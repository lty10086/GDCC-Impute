# GDCC-Impute
A graph-based conditional generative adversarial network for scRNA-seq data imputation.

## 📖 Overview
GDCC (Graph-based Conditional GAN) is designed to address the dropout events in single-cell RNA sequencing (scRNA-seq) data. By employing graph convolutional networks (GCN) to capture cell-cell relationships and a conditional GAN architecture, it effectively imputes missing values while preserving biological variance and structure.
![Fig1](https://github.com/user-attachments/assets/ddd220a7-bc47-400a-bce3-371a02ad74b2)

## 🚀 Quick Start

### 1. Installation
Ensure you have Python 3.10 and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Input data and format
sc_dataset: .csv format count scRNA-seq data, organized by cell (rows) and by gene (columns).

### 3. Run Example
```bash
Run the Dataset1 with 50% dropout eg: python Impute.py --datasets Dataset1/50% --target_sum 1e3 --resolution 0.3 --resolution 0.9 --g_lr 2e-4
```
```bash
Run the Kolodziejczyk with 30% dropout eg: python Impute.py --datasets Kolodziejczyk --target_sum 1e3 --resolution 0.3 --resolution 0.9 --g_lr 2e-2
```
Note: The GSE75748 (Time) dataset is for trajectory analysis and does not involve manual dropout simulation. You can directly use the raw data as drop_data.csv.

### 4. Look for each parameter of GDCC via
```bash
python Impute.py --help
```
```bash
options:
  -h, --help            show this help message and exit
  --datasets DATASETS [DATASETS ...]
                        Datasets to process (default: Need Input)
  --data_dir DATA_DIR   Directory containing dataset folders (default: ./dataset)
  --output_dir OUTPUT_DIR
                        Directory to save results (default: ./results)
  --K_param K_PARAM     [Key Hyperparameter] Number of nearest neighbors (K) for adjacency matrix. Recommended: [1, 10]. Default: 5
  --sub_clusters SUB_CLUSTERS
                        [Key Hyperparameter] Number of sub-clusters (Ks) for hierarchical clustering. Recommended: [1, 5]. Default: 2
  --seed SEED           Random seed (default: 100)
  --target_sum TARGET_SUM
                        Normalization target sum (default: 1000)
  --resolution RESOLUTION
                        Leiden resolution (default: 0.3)
  --latent_dim LATENT_DIM
                        Latent space dimension (default: 32)
  --dropout_thr DROPOUT_THR
                        Dropout identification threshold (default: 0.9)
  --gp_weight GP_WEIGHT
                        Gradient penalty weight (default: 1.0)
  --g_lr G_LR           Generator learning rate. Tip: Use 2e-2 for high-expression datasets. Default: 2e-4
  --g_epoch_ratio G_EPOCH_RATIO
                        Generator training steps per Discriminator step (G/D ratio). Default: 3
  --d_epoch_ratio D_EPOCH_RATIO
                        Discriminator training steps. Default: 1
  --num_epochs NUM_EPOCHS
                        Total training epochs. Recommended >= 600. Default: 600
```

### Parameter Description

| Datasets | target_sum | resolution | dropout_thr | g_lr |
| :--- | :--- | :---: | :--- | :--- |
| Dataset1-4 | 1e3 | 0.3 | 0.9 | 2e-4|
| Kolodziejczyk | 1e3 | 0.3 | 0.9 | 2e-2|
| Klein | 1e4 | 0.5 | 0.9 | 2e-4|
| Zeisel | 1e4 | 0.5 | 0.9 | 2e-4|
| Type | 1e3 | 0.3 | 0.3 | 2e-3|
| Time | 1e3 | 0.3 | 0.9 | 2e-3|
| Baron | 1e4 | 0.1 | 0.3 | 2e-3|
| Villani | 1e4 | 0.1 | 0.3 | 2e-3|
