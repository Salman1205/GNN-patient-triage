# Patient Triage Classification using Graph Neural Networks

This repository reproduces the methodology from the research paper:

**Defilippo, A., Veltri, P., Lió, P. et al. Leveraging graph neural networks for supporting automatic triage of patients. *Sci Rep* **14**, 12548 (2024).**  
https://doi.org/10.1038/s41598-024-63376-2

## Overview

This project implements the paper's patient similarity graph pipeline end‑to‑end inside `patient_triage_gnn.ipynb`. The notebook reproduces the GraphSAGE architecture highlighted in the paper, including the preprocessing safeguards (train/test split before scaling, SMOTE on the train set only, and Min-Max normalization) and the cosine-similarity threshold sweep (0.98 → 0.90) used to justify the final graph.

## Methodology

1. **Data preprocessing without leakage**  
   - Remove rows with missing triage labels.  
   - Train/test split (70/30 stratified) before any transforms.  
   - Mode imputation → SMOTE balancing → Min-Max scaling applied on train data, then reused for test.
2. **Graph construction**  
   - Build patient similarity graphs with cosine similarity.  
   - Threshold sweep (0.98 → 0.90) to confirm the paper’s preferred value (0.95).  
   - Graph statistics (edge count, degree distribution) logged for transparency.
3. **Graph Neural Networks**  
   - Paper-accurate GraphSAGE (5 SAGEConv layers, dropout 0.2) as the primary model.  
   - Reference implementations for GCN and GAT are included for experimentation.
4. **Evaluation**  
   - Train/validation curves, early stopping, and ReduceLROnPlateau scheduler.  
   - Hold-out test set predictions computed by linking test nodes to their top‑k train neighbors (cosine).  
   - Baseline comparisons with Random Forest and Logistic Regression.

## Dataset

The dataset (`archive/patient_priority.csv`) contains patient information including:
- Demographic features (age, gender)
- Clinical features (blood pressure, cholesterol, heart rate, etc.)
- Triage labels (red, orange, yellow)

## Installation

1. Clone this repository.
2. Create/activate a Python 3.10+ environment.
3. Install the dependencies used in the notebook (CUDA wheels optional):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install imbalanced-learn
   pip install torch-geometric
   pip install -r requirements.txt  # contains the remaining utilities
   ```

## Running the notebook

```bash
jupyter notebook patient_triage_gnn.ipynb
```

Inside the notebook you will:
- Inspect the dataset, distributions, and missing values.
- Execute the three preprocessing stages (imputation, SMOTE, scaling).
- Build cosine similarity graphs and visualize their properties.
- Train GraphSAGE for 200 epochs with Adam (lr=0.01, weight_decay=5e-4) on GPU if available.
- Evaluate on the isolated 30% test split plus baseline classifiers.

## Results (sample run)

The reference run included in the notebook reports:
- **GraphSAGE (cosine threshold 0.95)**: Accuracy 0.8896, Precision 0.9091, Recall 0.8896, F1 0.8977.
- **Random Forest baseline**: Accuracy 0.9791 (strict regularization to reduce overfitting).
- **Logistic Regression baseline**: Accuracy 0.8952.

Confusion matrices, classification reports, and metric tables generated in later cells provide additional detail.

## Repository contents

- `patient_triage_gnn.ipynb` – full reproduction, including graph sweep, training, and evaluation.
- `archive/patient_priority.csv` – emergency department dataset (features + triage labels).
- `requirements.txt` – base Python dependencies used in addition to the pip installs above.

## Requirements

- Python 3.10+
- PyTorch 2.5.1+ (CUDA 12.1 build confirmed) and Torch Geometric 2.7.0
- NumPy, pandas, scikit-learn, seaborn, matplotlib, imbalanced-learn, networkx
- See the notebook for the exact versions logged during execution

## Citation

If you use this code, please cite the original paper:

```
@article{defilippo2024leveraging,
  title={Leveraging graph neural networks for supporting automatic triage of patients},
  author={Defilippo, Annamaria and Veltri, Pierangelo and Li{\'o}, Pietro and Guzzi, Pietro Hiram},
  journal={Scientific Reports},
  volume={14},
  number={1},
  pages={12548},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

