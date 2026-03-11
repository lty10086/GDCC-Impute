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

### 2. Input data and format
sc_dataset: ‘.csv’ format count scRNA-seq data, by cell, by gene.

### 3. Run Example
```bash
python Impute.py --datasets ZINB --num_epochs 10 --K_param 5

For help on all available arguments:
```bash
python Impute.py --help
