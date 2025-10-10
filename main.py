import torch
import torch.nn as nn
import torchvision
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import torchmetrics
from tqdm import tqdm
from torchinfo import summary
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from utils import SplitDataset, FocalLoss
from augmentation import tf, mixup_data, cutmix_data
import random
import argparse

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    
def train(model, weights, transform_resize, embed_dim, local_model_path, root, index_cache, splits_file, get_splits, num_classes, epochs, accelerator, model_name):
    # Set environment variables
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_LOGS"] = "+dynamo"
    os.environ["TORCHDYNAMO_VERBOSE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    torch._dynamo.config.suppress_errors = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Version:", torch.version.cuda)
    print("cuDNN Version:", torch.backends.cudnn.version())
        
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    np.random.seed(seed)
    random.seed(seed)

    device = 'cuda' if accelerator else 'cpu'
    model = model(weights=weights)
    
    # Freeze all parameters, then unfreeze layers 4 to 11 and layer norm
    for param in model.parameters():
        param.requires_grad = False
    for param in model.encoder.layers[4:12].parameters():
        param.requires_grad = True
    for param in model.encoder.ln.parameters():
        param.requires_grad = True

    model.heads = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(embed_dim, num_classes)
    )
    
    # Optionally load a checkpoint
    if local_model_path:
        model.load_state_dict(torch.load(local_model_path, map_location=device))
        
    print(summary(model=model, input_size=(1, 3, 384, 384), 
                  col_names=["input_size", "output_size", "num_params", "trainable"],
                  col_width=20, row_settings=["var_names"]))
    
    model = model.to(device)
    print(f"Model loaded to {device}")
    
    train_tf, test_tf = tf(transform_resize)
    
    # Load Datasets 
    ds = SplitDataset(root=root, index_cache=index_cache, splits_file=splits_file)
        
    train_data = ds.get_split(get_splits[0], transform=train_tf)
    val_data   = ds.get_split(get_splits[1], transform=test_tf)    
    
    print("Number of classes in dataset:", len(ds.dataset.class_to_idx), ds.dataset.class_to_idx)
    print("num_classes argument:", num_classes)
    
    print(f"Full Train data size (before sampling): {len(train_data)}")
    print(f"Full Validation data size (before sampling): {len(val_data)}")
       
    # adding minor classes to train_data as dataset is imbalance 
    minority_class_ids = {1, 3, 5, 13}
    original_train_indices = train_data.indices
    final_train_indices = []
    subsample_rate = 20

    for i, original_idx in enumerate(original_train_indices):
        _, label = ds.dataset.samples[original_idx]

        if label in minority_class_ids:
            final_train_indices.append(original_idx)
        elif i % subsample_rate == 0 and label not in minority_class_ids:
            final_train_indices.append(original_idx)

    train_dataset = torch.utils.data.Subset(ds.dataset, final_train_indices)
    train_dataset.dataset.transform = train_tf
    val_indices = list(range(0, len(val_data), 5)) 
    val_dataset = torch.utils.data.Subset(val_data, val_indices)
    
    print(f'Train data, size: {len(train_dataset)}')
    print(f'Test data, size: {len(val_dataset)}')

    g = torch.Generator()
    g.manual_seed(42)
    
    # DataLoaders 
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                                   shuffle=True, 
                                                   num_workers=4,
                                                   worker_init_fn=seed_worker, generator=g)
    val_dataloader  = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
 
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)   
    Scaler = torch.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=0.01, threshold_mode='rel' ,verbose=True, eps=1e-11)


    # Loss Function 
    targets = [ds.dataset.targets[i] for i in train_dataset.indices]
    counts = Counter(targets)
    total = sum(counts.values())
    class_weights = [total / max(1, counts.get(i, 0)) for i in range(num_classes)]
    weights_tensor = torch.tensor(class_weights).to(device)
    
    gamma_value = 2.0 
    label_smoothing_value = 0.1
    loss_function = FocalLoss(gamma=gamma_value, 
                            alpha=weights_tensor, 
                            # alpha=None, 
                            reduction='mean', 
                            label_smoothing=label_smoothing_value)    
    # Metrics 
    top1_acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, top_k=1).to(device)
    top5_acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(device)
    
    train_accs_top1, train_accs_top5 = [], []
    test_accs_top1, test_accs_top5 = [], []
    train_losses, test_losses = [], []
    start_time = time.time()
    
    mixup_alpha = 0.05  
    cutmix_alpha = 0.1 
    mixup_prob = 0.5
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch} training'):
            data, targets = batch[0].to(device), batch[1].to(device)
                        
            targets = targets.long() 
            
            if np.random.rand() < mixup_prob:
                # Apply MixUp
                data, targets_a, targets_b, lam = mixup_data(data, targets, alpha=mixup_alpha, device=device)
            else:
                # Apply CutMix
                data, targets_a, targets_b, lam = cutmix_data(data, targets, alpha=cutmix_alpha, device=device)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(data)
                loss_a = loss_function(outputs, targets_a)
                loss_b = loss_function(outputs, targets_b)
                loss = lam * loss_a + (1 - lam) * loss_b
                # loss = loss_function(outputs, targets)

            Scaler.scale(loss).backward()
            Scaler.step(optimizer)
            Scaler.update()
            
            total_train_loss += loss.item()
            top1_acc_metric.update(outputs, targets)
            top5_acc_metric.update(outputs, targets)
        
        train_accs_top1.append(top1_acc_metric.compute().item())
        train_accs_top5.append(top5_acc_metric.compute().item())
        train_losses.append(total_train_loss / len(train_dataloader))
        top1_acc_metric.reset()
        top5_acc_metric.reset()
        
        model.eval()
        total_test_loss = 0
        with torch.inference_mode():
            for batch in tqdm(val_dataloader, desc=f'Epoch {epoch} testing'):
                data, targets = batch[0].to(device), batch[1].to(device)
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(data)
                    loss = loss_function(outputs, targets)
                total_test_loss += loss.item()
                top1_acc_metric.update(outputs, targets)
                top5_acc_metric.update(outputs, targets)

        test_accs_top1.append(top1_acc_metric.compute().item()) 
        test_accs_top5.append(top5_acc_metric.compute().item())
        test_losses.append(total_test_loss / len(val_dataloader))
        top1_acc_metric.reset()
        top5_acc_metric.reset()
        
        
        end_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Train Loss {train_losses[-1]:.4f}, Train Top-1 Acc {train_accs_top1[-1]:.4f}%, Train Top-5 Acc {train_accs_top5[-1]:.4f}%, "
              f"Test Loss {test_losses[-1]:.4f}, Test Top-1 Acc {test_accs_top1[-1]:.4f}%, Test Top-5 Acc {test_accs_top5[-1]:.4f}%, Time {end_time/60:.2f} min, LR {current_lr}")
        
        scheduler.step(test_losses[-1])
        torch.save(model.state_dict(), f'./{model_name}_results/{model_name}_{epoch}.pth')
        
    # Plot Results 
    plt.style.use('ggplot')
    epochs_range = np.arange(1, epochs + 1)
    plt.figure()
    plt.plot(epochs_range, train_accs_top1, label='Train Top-1 Accuracy')
    plt.plot(epochs_range, test_accs_top1, label='Test Top-1 Accuracy')
    plt.plot(epochs_range, train_accs_top5, label='Train Top-5 Accuracy')
    plt.plot(epochs_range, test_accs_top5, label='Test Top-5 Accuracy')
    plt.legend()
    plt.title("Accuracy over Epochs")
    plt.show()
    
    plt.figure()
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, test_losses, label='Test Loss')
    plt.legend()
    plt.title("Loss over Epochs")
    plt.show()
    
    print(f'Best Test Top-1 Accuracy: {max(test_accs_top1):.3f}%')
    print(f'Best Test Top-5 Accuracy: {max(test_accs_top5):.3f}%')
    
    return test_accs_top1, test_accs_top5

    
def test(model, weights, transform_resize, embed_dim, local_model_path, root, index_cache, splits_file, get_splits, num_classes, accelerator, mapping, model_name):
    # Set environment variables
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_LOGS"] = "+dynamo"
    os.environ["TORCHDYNAMO_VERBOSE"] = "1"
    torch._dynamo.config.suppress_errors = True

    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Version:", torch.version.cuda)
    print("cuDNN Version:", torch.backends.cudnn.version())

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    device = 'cuda' if accelerator else 'cpu'
    model = model(weights=weights)
    
    model.heads = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(embed_dim, num_classes))
        
    for param in model.parameters():
        param.requires_grad = False
    
    # Optionally load a checkpoint
    if local_model_path:
        model.load_state_dict(torch.load(local_model_path, map_location=device))
        
    print(summary(model=model, input_size=(1, 3, 384, 384), 
                  col_names=["input_size", "output_size", "num_params", "trainable"],
                  col_width=20, row_settings=["var_names"]))
    
    model = model.to(device)
    print(f"Model loaded to {device}")
    
    train_tf, test_tf = tf(transform_resize)
    
    # Load Datasets 
    ds = SplitDataset(root=root, index_cache=index_cache, splits_file=splits_file)
    test_data = ds.get_split(get_splits, transform=test_tf)
    
    print("Number of classes in dataset:", len(ds.dataset.class_to_idx), ds.dataset.class_to_idx)
    print("num_classes argument:", num_classes)
    print(f"Full Train data size: {len(test_data)}")
    
    # DataLoaders 
    test_dataloader  = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)
    
    # Loss Function 
    loss_function = nn.CrossEntropyLoss()
    
    # Metrics 
    top1_acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, top_k=1).to(device)
    top5_acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(device)
    
    model.eval()
    total_test_loss = 0
    
    # For confusion matrix and classification report
    all_preds = []
    all_targets = []
    
    start_time = time.time()
        
    with torch.inference_mode():
        for batch in tqdm(test_dataloader, desc='testing'):
            data, targets = batch[0].to(device), batch[1].to(device)

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(data)  

                loss = loss_function(outputs, targets)

            total_test_loss += loss.item()

            top1_acc_metric.update(outputs, targets)
            top5_acc_metric.update(outputs, targets)

            # Store predictions and targets for confusion matrix
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Compute metrics
    test_top1_acc = top1_acc_metric.compute().item()
    test_top5_acc = top5_acc_metric.compute().item()
    avg_test_loss = total_test_loss / len(test_dataloader)
    
    times = time.time() - start_time

    print(f'Test Loss: {avg_test_loss:.4f}, Test Top-1 Acc: {test_top1_acc:.4f}, Test Top-5 Acc: {test_top5_acc:.4f}, Time {times/60:.2f} min')

    with open(mapping, 'r') as f:
        class_mappings = f.readlines()
    class_names = [line.strip().split(' ', 1)[1] for line in class_mappings]
    
    # Classification report
    class_report = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    print("\nClassification Report:\n")
    print(class_report)
    
    class_report = classification_report(all_targets, all_preds, target_names=class_names, digits=4, output_dict=True) 
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')    
    plt.tight_layout() 
    plt.savefig(f'{model_name}_results/confusion_matrix.png')
    plt.show()
    plt.close() 
    
    return class_report, conf_matrix


### Implemented for jupyter
# def loop(root, index_cache, splits_file, get_splits, epochs, Is_train=True, local_model_path=False, model_name=False, mapping=False):

#     model_fn = torchvision.models.vit_b_16
#     weights_enum = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1

#     if Is_train:
#         test_accs_top1, test_accs_top5 = train(model=model_fn,
#             weights=weights_enum,
#             transform_resize=384,
#             embed_dim=768,
#             local_model_path=False,  
#             root=root, 
#             index_cache=index_cache, 
#             splits_file=splits_file,
#             get_splits=get_splits,
#             num_classes=19,
#             epochs=epochs,
#             accelerator=True,
#             model_name=model_name)
        
#         return test_accs_top1, test_accs_top5
    
#     else:
#         class_report, conf_matrix = test(model=model_fn,
#             weights=weights_enum,
#             transform_resize=384,
#             embed_dim=768,
#             local_model_path=local_model_path,
#             root=root,
#             index_cache=index_cache,
#             splits_file=splits_file,
#             get_splits=get_splits,
#             num_classes=19,
#             accelerator=True,
#             model_name=model_name,
#             mapping=mapping)
        
#         return class_report, conf_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test a ViT on the 50Salads dataset.")
    parser.add_argument("--split", type=int, required=True, help="The cross-validation split number to use (e.g., 1, 2, 3, 4, 5).")
    parser.add_argument("--mode", type=str, required=True, choices=['train', 'test'], help="Set the script to 'train' or 'test' mode.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train for (used in 'train' mode).")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the model checkpoint .pth file (required for 'test' mode).")
    parser.add_argument("--root", type=str, default="frames", help="Directory with all the video frames.")
    parser.add_argument("--index_cache", type=str, default="frames_index.txt", help="Path to the frame index cache file.")
    parser.add_argument("--splits_file", type=str, default="splits.pkl", help="Path to the splits pickle file.")
    parser.add_argument("--mapping", type=str, default="./mapping.txt", help="Path to the class mapping file.")
    
    args = parser.parse_args()

    model_fn = torchvision.models.vit_b_16
    weights_enum = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    model_name = f'Split{args.split}'

    if args.mode == 'train':
        print(f"--- Running in TRAIN mode for Split {args.split} ---")
        get_splits = [f"split{args.split}_train", f"split{args.split}_test"]
        
        result_path = f'./Split{args.split}_results'
        os.makedirs(result_path, exist_ok=True)

        train(model=model_fn,
              weights=weights_enum,
              transform_resize=384,
              embed_dim=768,
              local_model_path=False,  
              root=args.root, 
              index_cache=args.index_cache, 
              splits_file=args.splits_file,
              get_splits=get_splits,
              num_classes=19,
              epochs=args.epochs,
              accelerator=True,
              model_name=model_name)
    
    elif args.mode == 'test':
        print(f"--- Running in TEST mode for Split {args.split} ---")
        if not args.checkpoint_path:
            raise ValueError("--checkpoint_path is required for 'test' mode.")
        
        get_splits = f"split{args.split}_test"

        test(model=model_fn,
             weights=weights_enum,
             transform_resize=384,
             embed_dim=768,
             local_model_path=args.checkpoint_path,
             root=args.root,
             index_cache=args.index_cache,
             splits_file=args.splits_file,
             get_splits=get_splits,
             num_classes=19,
             accelerator=True,
             model_name=model_name,
             mapping=args.mapping)