from torchvision.datasets import ImageFolder
import cv2
import os 
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, Subset
from pathlib import Path
import glob 
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

INDEX_CACHE_FILE = "frames_index.txt"

def extract_frames_from_videos(rgb_dir, frames_dir, ground_truth_dir, mapping_file, target_fps=30, frame_size=(640, 480)):
  """
  Extract frames from all .avi videos in a folder.
  """
  rgb_dir = Path(rgb_dir)
  
  frames_dir = Path(frames_dir)
  frames_dir.mkdir(parents=True, exist_ok=True)

  mapping_file = Path(mapping_file) 
  class_mapping = {}
  with open(mapping_file, "r") as f:
    for line in f:
      class_id, class_name = line.strip().split(" ", 1)
      class_mapping[class_name] = class_name 
    
  ground_truth_dir = Path(ground_truth_dir) 

  video_files = list(rgb_dir.glob("*.avi"))

  for video_path in video_files:
    video_name = video_path.stem
    ground_truth_file = ground_truth_dir / f"{video_name}.txt"
    with open(ground_truth_file, "r") as f:
      labels = f.read().splitlines()
      
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
      print(f"Could not open {video_path}")
      continue

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(original_fps / target_fps)) if original_fps > target_fps else 1

    frame_count = 0
    saved_count = 0
    
    print(f"Extracting frames from {video_name}...")

    while True:
      # Check for label count BEFORE trying to read the frame/label
      if frame_count >= len(labels):
        if saved_count < len(labels):
          pass
        else:
          print(f"More frames than labels in {video_name}, stopping extraction.")
        break
        
      ret, frame = cap.read()
      if not ret:
        break

      if frame_count % frame_interval == 0:
        label = labels[frame_count]
        
        frame_resized = cv2.resize(frame, frame_size)
        frame_filename = f"{video_name}_frame_{saved_count}.jpg"
        
        # Check if the label is valid before writing
        if label in class_mapping:
          cv2.imwrite(str(frames_dir / frame_filename), frame_resized)
          saved_count += 1
        else:
          print(f"Warning: Frame {frame_count} in {video_name} has unknown label: {label}. Skipping.")


      frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames from {video_name}")

  print("Done extracting all videos!")


def dump_splits():
  """Load train and test splits into a dict and save as pickle."""
  splits = {}
  for i in range(5): 
    for split in ["train", "test"]: 
      # Using glob to find the splits directory structure robustly
      split_files = glob.glob(f"splits/{split}/{split}.split{i}.bundle")
      if not split_files:
        print(f"Warning: Split file for {split}.split{i}.bundle not found.")
        continue
        
      split_file = Path(split_files[0])
      with open(split_file, "r") as f:
        videos = [line.strip() for line in f]
      
      # The filenames in the bundle are 'rgb-XX-X.txt', we need the stem 'rgb-XX-X'
      splits[f"split{i}_{split}"] = [Path(v).stem for v in videos]

  with open("splits.pkl", "wb") as f:
    pickle.dump(splits, f)

  print("Saved splits.pkl with keys:", list(splits.keys()))
  
   
class FrameDataset(Dataset):
  """
  This class acts as a minimal PyTorch Dataset, initialized with the final list of (path, target) tuples.
  """
  
  def __init__(self, samples, class_to_idx, transform=None, loader=default_loader):
    self.samples = samples
    self.targets = [s[1] for s in samples] 
    self.class_to_idx = class_to_idx
    self.transform = transform
    self.loader = loader

    # Manually set the classes list from the provided mapping for compatibility
    sorted_items = sorted(class_to_idx.items(), key=lambda item: item[1])
    self.classes = [item[0] for item in sorted_items] 
    
  def __getitem__(self, index):
    path, target = self.samples[index]
    sample = self.loader(str(path)) 
    if self.transform is not None:
      sample = self.transform(sample)
    return sample, target

  def __len__(self):
    return len(self.samples)


class SplitDataset:
  def __init__(self, root, ground_truth_dir="groundTruth", index_cache=INDEX_CACHE_FILE, splits_file="splits.pkl", mapping_file="mapping.txt"):
    """
    Initializes the dataset manager.
    """
    self.root = Path(root)
    self.index_cache = Path(index_cache) 
    self.splits_file = Path(splits_file)
    
    # Load the official 19 action class mapping
    self.class_to_idx = {}
    if not os.path.exists(mapping_file):
      raise FileNotFoundError(f"Mapping file not found at {mapping_file}")
      
    with open(mapping_file, "r") as f:
      for line in f:
        try:
          # class_id is 0-indexed integer, class_name is the string label
          class_id, class_name = line.strip().split(" ", 1)
          self.class_to_idx[class_name] = int(class_id)
        except ValueError:
          continue

    correct_num_classes = len(self.class_to_idx)
    
    # Build or load FrameDataset samples + cached index 
    if self.index_cache.exists():
      print(f"Loading samples from cache: {self.index_cache}")
      samples = []
      
      # Use 'latin-1' encoding and read line by line for robustness
      with open(self.index_cache, 'r', encoding='latin-1') as f:
        # Generator to filter NUL bytes ('\x00') and yield clean lines
        clean_lines = (line.replace('\x00', '').strip() for line in f)

        for line in clean_lines:
          if not line: continue 

          parts = line.split(',', 1) # Split only once to handle paths with commas if they exist
          if len(parts) != 2:
            print(f"Warning: Skipping malformed row in index cache: {line}")
            continue

          path_str, class_id_str = parts[0], parts[1]

          try:
            action_id = int(class_id_str)
            if action_id < correct_num_classes:
              samples.append((path_str, action_id))
          except ValueError:
            print(f"Warning: Skipping sample with non-integer class ID: {class_id_str}")
            continue
      
    else:
      print("Cache not found. Aggregating samples from frames/ and groundTruth/ (This may take a while)...")
      all_samples = []
      
      ground_truth_path = Path(ground_truth_dir)
      video_files = list(ground_truth_path.glob("*.txt"))
      
      if not video_files:
        raise RuntimeError(f"No ground truth files found in {ground_truth_path}.")

      total_frames_indexed = 0
      
      for gt_file in video_files:
        video_name = gt_file.stem # e.g., 'rgb-01-1'
        
        with open(gt_file, "r") as f:
          labels = f.read().splitlines()
        
        for i, label_name in enumerate(labels):
          frame_filename = f"{video_name}_frame_{i}.jpg" 
          frame_path = self.root / frame_filename
          
          if frame_path.exists():
            action_id = self.class_to_idx.get(label_name)
            
            if action_id is not None:
              all_samples.append((frame_path, action_id))
              total_frames_indexed += 1

      samples = all_samples
      
      if not samples:
        raise RuntimeError("No frames were successfully collected. Check your data paths and extraction step.")
      
      print(f"Finished scraping {total_frames_indexed:,} frames. Saving index to cache: {self.index_cache}.")
      
      with open(self.index_cache, 'w') as f:
        for path, id in samples:
          # Write each frame index entry in "path,id\n" format
          f.write(f"{os.path.abspath(path)},{id}\n")

    self.dataset = FrameDataset(
      samples=samples, 
      class_to_idx=self.class_to_idx
    )
    print(f"\n--- DEBUG: Total samples loaded into FrameDataset: {len(self.dataset.samples)} ---")

    # Load split definitions 
    try:
      with open(self.splits_file, "rb") as f:
        self.splits = pickle.load(f)
    except FileNotFoundError:
      raise FileNotFoundError(f"Splits file not found at {self.splits_file}. Please ensure the file exists.")


  def get_split(self, split_id, transform=None):
    """
    Return a Subset of the dataset corresponding to a specific split.
    Args:
      split_id (str): Key of the split (e.g., "split1_train", "split1_test").
      transform (callable, optional): Optional transform to be applied to the Subset.
    """
    if split_id not in self.splits:
      raise ValueError(f"Split {split_id} not found! Available: {list(self.splits.keys())}")

    allowed_videos = set(self.splits[split_id]) # e.g., 'rgb-01-1'
    
    # Filter the dataset samples to include only frames belonging to the allowed videos
    indices = [
      i for i, (path, _) in enumerate(self.dataset.samples)
      if any(v in path for v in allowed_videos)
    ]

    if transform is not None:
      self.dataset.transform = transform
    
    return Subset(self.dataset, indices)


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalance multi-class classification, based on:
    'Focal Loss for Dense Object Detection' by Lin et al.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', label_smoothing=0.0):
        """
        Args:
            gamma (float): Focusing parameter. Typically gamma=2.0.
            alpha (Tensor, optional): Class-wise weighting factor (alpha_t). 
                                     If None, no class balancing is applied.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            label_smoothing (float): Label smoothing parameter.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        """
        Args:
            input (Tensor): Logits of shape (N, C), where N is batch size, C is number of classes.
            target (Tensor): Ground truth class indices of shape (N).
        """
        # Standard Cross Entropy Loss computation
        ce_loss = F.cross_entropy(input, target, reduction='none', 
                                  weight=None, 
                                  label_smoothing=self.label_smoothing)

        # Convert logits to probabilities: p = exp(input) / sum(exp(input))
        p = torch.exp(-ce_loss) # p = softmax(input) -> p_t. Since -log(p_t) = ce_loss, p_t = exp(-ce_loss).
                                
        # Calculate the focusing factor (1 - p_t)^gamma
        focal_term = (1.0 - p) ** self.gamma
        loss = focal_term * ce_loss
        
        # Apply the optional class-wise alpha factor (alpha_t)
        if self.alpha is not None:
            # target is LongTensor of class indices. alpha is Tensor of shape (C).
            # alpha_t for each sample
            alpha_t = self.alpha.gather(0, target) 
            loss = alpha_t * loss

        # 6. Apply final reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else: # 'none'
            return loss
          
          
          
extract_frames_from_videos(
  rgb_dir="rgb",   
  frames_dir="frames",
  ground_truth_dir="groundTruth",
  mapping_file = "mapping.txt",
  target_fps=30,      
  frame_size=(640, 480)   
)  

dump_splits()

root = "frames"
splits_file="splits.pkl"

SplitDataset(root=root, index_cache=INDEX_CACHE_FILE, splits_file=splits_file, mapping_file="mapping.txt")
