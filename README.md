# Action Recognition on the 50Salads Dataset/ 82.7 ACC

![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This repository contains the code for fine-tuning a pre-trained Vision Transformer (ViT-B/16) model for fine-grained action recognition on the **50Salads** dataset. The project demonstrates several techniques for handling significant class imbalance, including a manual 
oversampling strategy and the use of Focal Loss, only by training `30` percent of the data.

The Demo is Available on [Huggigface](https://huggingface.co/spaces/Factor054/Salads_Classifier_Demo).

## Dataset: 50Salads

The `50Salads` dataset is a popular benchmark for fine-grained action recognition. It consists of 50 videos of individuals preparing mixed salads. The actions are annotated at the frame level, leading to a challenging classification problem with significant variation in class duration and frequency.

As shown in `class_distribution.txt`, the dataset suffers from a severe class imbalance, with classes like `peel_cucumber` having over 61,000 frames while `place_cheese_into_bowl` has only around 10,000. This project places a strong emphasis on strategies to mitigate this imbalance.
## Results
<table class="dataframe results-table" border="0">
  <thead>
    <tr style="text-align: right;">
      <th>Class</th>
      <th>Mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cut_tomato</th>
      <td>94.38%</td>
    </tr>
    <tr>
      <th>place_tomato_into_bowl</th>
      <td>51.03%</td>
    </tr>
    <tr>
      <th>cut_cheese</th>
      <td>88.01%</td>
    </tr>
    <tr>
      <th>place_cheese_into_bowl</th>
      <td>63.39%</td>
    </tr>
    <tr>
      <th>cut_lettuce</th>
      <td>83.40%</td>
    </tr>
    <tr>
      <th>place_lettuce_into_bowl</th>
      <td>43.77%</td>
    </tr>
    <tr>
      <th>add_salt</th>
      <td>82.28%</td>
    </tr>
    <tr>
      <th>add_vinegar</th>
      <td>90.77%</td>
    </tr>
    <tr>
      <th>add_oil</th>
      <td>88.12%</td>
    </tr>
    <tr>
      <th>add_pepper</th>
      <td>70.37%</td>
    </tr>
    <tr>
      <th>mix_dressing</th>
      <td>58.33%</td>
    </tr>
    <tr>
      <th>peel_cucumber</th>
      <td>90.62%</td>
    </tr>
    <tr>
      <th>cut_cucumber</th>
      <td>84.35%</td>
    </tr>
    <tr>
      <th>place_cucumber_into_bowl</th>
      <td>49.45%</td>
    </tr>
    <tr>
      <th>add_dressing</th>
      <td>75.29%</td>
    </tr>
    <tr>
      <th>mix_ingredients</th>
      <td>65.83%</td>
    </tr>
    <tr>
      <th>serve_salad_onto_plate</th>
      <td>91.77%</td>
    </tr>
    <tr>
      <th>action_start</th>
      <td>93.03%</td>
    </tr>
    <tr>
      <th>action_end</th>
      <td>94.57%</td>
    </tr>
    <tr class="overall-accuracy">
      <th>MEAN</th>
      <td>82.71%</td>
    </tr>
  </tbody>
</table>

## How to Use
This project uses the five standard cross-validation splits for the 50Salads dataset.

> **Note:** The `main.py` script has been refactored to accept command-line arguments. The original method of running via Jupyter Notebooks (`ViT_Split1_FT.ipynb`, etc.) is still possible but command-line usage is fully supported.

### 1. Setup

**Prerequisites:**
* Python 3.10+
* PyTorch with CUDA support
* `pip` for package management

**Installation:**

1.  Clone the repository:
    ```bash
    git clone https://github.com/MrKGhasemi/50salads
    cd 50salads
    ```

2.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### 2. Data Preparation
   
1.  **Dataset:** Download the 50Salads dataset and place the `rgb` videos, `groundTruth` annotations, `mapping.txt` file, and `splits` in the root directory.
[download](https://drive.google.com/file/d/1SnoAsUzNcHFykRIpCRc4F1L-mIIxpVP0/view?usp=sharing) the dataset

2.  **Extract Frames:** The `utils.py` script contains a function `extract_frames_from_videos` that can be used to process the `.avi` files into a `frames/` directory. Also, generate the `frames_index.txt` cache file and the `splits.pkl` file from the raw dataset files.
- prepare data by:
    ```bash
    python utils.py
    ```
    
### 3. Training

To train the model on a specific split (e.g., `split1`), run the `main.py` script in `train` mode. The script will handle the manual oversampling, apply augmentations, and save the model checkpoints in the corresponding `SplitX_results/` directory.

   ```bash
    python main.py --split 1 --epochs 30 --mode train
   ```

### 4. Testing
To evaluate a trained model, run the script in test mode, providing the path to the model checkpoint. This will generate a classification report and a confusion matrix.

```bash
python main.py --split 1 --mode test --checkpoint_path "Split1_results/Split1_29.pth"
```

