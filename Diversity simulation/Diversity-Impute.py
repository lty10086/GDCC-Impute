import torch
import numpy as np
import pandas as pd
import random
import os
import argparse
import warnings
import matplotlib.pyplot as plt

try:
    from GDCC import (
        Generator, Discriminator, 
        preprocessing, cluster, identify_dropout, 
        compute_labels_relation, A_lap_norm, 
        train_DCImpute, imp_example
    )
except ImportError as e:
    print(f"Error importing GDCC: {e}")
    print("Please ensure 'GDCC.py' is in the same directory as 'Impute.py'.")
    exit(1)

import scanpy as sc
sc.set_figure_params(dpi=50, dpi_save=600, figsize=(4, 4))

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 辅助函数定义 (评估指标等)
# ---------------------------------------------------------

def purity_score(true_labels, pred_labels):
    """计算聚类纯度 (Purity Score)"""
    contingency = contingency_matrix(true_labels, pred_labels)
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)

def jaccard_autoclass(ytrue, ypred):
    """计算自定义的 Jaccard 指数"""
    n = len(ytrue)
    a, b, c, d = 0, 0, 0, 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if (ypred[i] == ypred[j]) and (ytrue[i] == ytrue[j]):
                a += 1
            elif (ypred[i] == ypred[j]) and (ytrue[i] != ytrue[j]):
                b += 1
            elif (ypred[i] != ypred[j]) and (ytrue[i] == ytrue[j]):
                c += 1
            else:
                d += 1
    
    if (a + b + c) == 0:
        return 0.0
    return a / (a + b + c)

def calculate_l1(imputed_data, original_data):
    return np.mean(np.abs(original_data - imputed_data))

def calculate_rmse(imputed_data, original_data):
    return np.sqrt(np.mean((original_data - imputed_data) ** 2))

def calculate_pearson(imputed_data, original_data):
    Y = original_data.reshape(-1)
    fake_Y = imputed_data.reshape(-1)
    
    fake_Y_mean, Y_mean = np.mean(fake_Y), np.mean(Y)
    
    numerator = np.sum((fake_Y - fake_Y_mean) * (Y - Y_mean))
    denominator = np.sqrt(np.sum((fake_Y - fake_Y_mean) ** 2)) * np.sqrt(np.sum((Y - Y_mean) ** 2))
    
    if denominator == 0:
        return 0.0
    return numerator / denominator

def norm_and_eval_view(data, label, title_name):
    """
    执行标准化、UMAP降维，并基于UMAP进行K-means聚类以计算评估指标
    """
    data_cells_genes = data.copy()
    
    obs = pd.DataFrame(index=data_cells_genes.index.astype(str))
    obs['label'] = label
    var = pd.DataFrame(index=data_cells_genes.columns.astype(str))
    X = data_cells_genes.values

    adata = sc.AnnData(X, obs=obs, var=var)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    norm_data = adata.X

    sc.pp.neighbors(adata, n_neighbors=5, n_pcs=40, use_rep='X')
    sc.tl.umap(adata)
    
    X_umap = adata.obsm['X_umap']
    
    K = len(np.unique(label))
    # 增加 n_init 以避免警告
    kmeans = KMeans(n_clusters=K, random_state=1, n_init=10).fit(X_umap)
    cluster_label = kmeans.labels_

    ari_val = np.round(adjusted_rand_score(label, cluster_label), 3)
    ji_val = np.round(jaccard_autoclass(label, cluster_label), 3)
    nmi_val = np.round(normalized_mutual_info_score(label, cluster_label), 3)
    ps_val = np.round(purity_score(label, cluster_label), 3)
    
    try:
        sw_val = np.round(silhouette_score(X_umap, label), 3)
    except ValueError:
        sw_val = 0.0
    
    return norm_data, ari_val, ji_val, nmi_val, ps_val, sw_val

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# ---------------------------------------------------------
# 主程序逻辑
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GDCC: Graph-based Conditional GAN for scRNA-seq Imputation")
    
    # --- 数据与路径参数 ---
    parser.add_argument('--datasets', type=str, nargs='+', default=['ZINB', 'NB', 'Mixture'],
                        help='Datasets to process (default: ZINB NB Mixture)')
    parser.add_argument('--data_dir', type=str, default='./dataset',
                        help='Directory containing dataset folders (default: ./dataset)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results (default: ./results)')
    
    # --- 核心超参数 ---
    # K: 邻接矩阵构建时的近邻数
    parser.add_argument('--K_param', type=int, default=5,
                        help='[Key Hyperparameter] Number of nearest neighbors (K) for adjacency matrix. Recommended: [1, 10]. Default: 5')
    # Ks: 层次聚类的子簇数
    parser.add_argument('--sub_clusters', type=int, default=2,
                        help='[Key Hyperparameter] Number of sub-clusters (Ks) for hierarchical clustering. Recommended: [1, 5]. Default: 2')
    
    # --- 模型训练参数 ---
    parser.add_argument('--seed', type=int, default=100, help='Random seed (default: 100)')
    parser.add_argument('--target_sum', type=float, default=1e3, help='Normalization target sum (default: 1000)')
    parser.add_argument('--resolution', type=float, default=0.3, help='Leiden resolution (default: 0.3)')
    parser.add_argument('--latent_dim', type=int, default=32, help='Latent space dimension (default: 32)')
    parser.add_argument('--dropout_thr', type=float, default=0.9, help='Dropout identification threshold (default: 0.9)')
    parser.add_argument('--gp_weight', type=float, default=1.0, help='Gradient penalty weight (default: 1.0)')
    
    # 学习率策略
    parser.add_argument('--g_lr', type=float, default=2e-4,
                        help='Generator learning rate. Tip: Use 2e-2 for high-expression datasets. Default: 2e-4')
    
    parser.add_argument('--g_epoch_ratio', type=int, default=3,
                        help='Generator training steps per Discriminator step (G/D ratio). Default: 3')
    parser.add_argument('--d_epoch_ratio', type=int, default=1,
                        help='Discriminator training steps. Default: 1')
    parser.add_argument('--num_epochs', type=int, default=600,
                        help='Total training epochs. Recommended >= 600. Default: 600')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("GDCC Imputation Pipeline Started")
    print("="*60)
    print(f"Configuration:")
    print(f"  Datasets: {args.datasets}")
    print(f"  Key Hyperparameters:")
    print(f"    - K (neighbors): {args.K_param} (Range: 1-10)")
    print(f"    - Ks (sub-clusters): {args.sub_clusters} (Range: 1-5)")
    print(f"  Training Params:")
    print(f"    - Epochs: {args.num_epochs}")
    print(f"    - G LR: {args.g_lr}")
    print(f"    - G/D Ratio: {args.g_epoch_ratio}:{args.d_epoch_ratio}")
    print("="*60)

    all_results = []

    for dataset in args.datasets:
        print(f"\n>>> Processing: {dataset}")
        
        set_seed(args.seed)
        
        data_path = os.path.join(args.data_dir, dataset)
        dropout_file = os.path.join(data_path, 'drop_data.csv')
        labels_file = os.path.join(data_path, 'labels.csv')
        true_data_file = os.path.join(data_path, 'true_data.csv')

        if not os.path.exists(dropout_file):
            print(f"  [SKIP] File not found: {dropout_file}")
            continue

        # 1. Load Data
        df_drop_data = pd.read_csv(dropout_file, index_col=0).T
        drop_data = df_drop_data.values
        ncell, ngene = drop_data.shape
        print(f"  Shape: {ncell} cells x {ngene} genes")

        # 2. Preprocessing & Clustering (Calls GDCC.py functions)
        norm_data, pca_matrix, leiden_labels, label_leiden_tensor, N_leiden = preprocessing(
            df_drop_data, args.target_sum, args.resolution
        )
        
        kmeans_labels_tensor = cluster(pca_matrix, N_leiden, 'kmeans')
        spectral_labels_tensor = cluster(pca_matrix, N_leiden, 'spectral')
        
        # 3. Identify Dropouts
        raw_data, mask_tensor = identify_dropout(leiden_labels, norm_data.T, drop_data, args.dropout_thr)
        
        # 4. Construct Conditional Matrices (Using K and Ks)
        Relation_matrix = compute_labels_relation(pca_matrix, leiden_labels, N_leiden, args.sub_clusters)
        CA_tensor = A_lap_norm(Relation_matrix, args.K_param)
        
        # 5. Initialize Models
        G = Generator(ngene, N_leiden, args.latent_dim, args.latent_dim)
        D = Discriminator(ngene, 1024, 256, 64, 16, N_leiden, args.latent_dim)
        
        # 6. Train
        print("  Training model...")
        losses, trained_generator = train_DCImpute(
            G, D, raw_data, CA_tensor, args.num_epochs, 
            label_leiden_tensor, mask_tensor, kmeans_labels_tensor, spectral_labels_tensor,
            args.latent_dim, args.gp_weight, args.g_lr, args.g_epoch_ratio, args.d_epoch_ratio
        )
        
        # 7. Impute
        print("  Generating imputed data...")
        imputed_data, generated_data = imp_example(
            label_leiden_tensor, CA_tensor, trained_generator, drop_data, mask_tensor, args.latent_dim
        )
        
        # Save Imputed Data
        df_Imp = pd.DataFrame(imputed_data, index=df_drop_data.index, columns=df_drop_data.columns)
        out_path = os.path.join(args.output_dir, f'GDCC_imputed_{dataset}.csv')
        df_Imp.T.to_csv(out_path)
        print(f"  Saved: {out_path}")

        # 8. Evaluation
        print("  Evaluating...")
        
        # Load Ground Truth Labels
        true_label = np.zeros(ncell)
        if os.path.exists(labels_file):
            cellinfo = pd.read_csv(labels_file)
            if 'x' in cellinfo.columns:
                Y = cellinfo['x'].values
                label_map = {"Group1": 0, "Group2": 1, "Group3": 2, "Group4": 3, "Group5": 4}
                true_label = np.array([label_map.get(val, 0) for val in Y])
        
        # Load True Data for Metrics
        true_norm_data = None
        if os.path.exists(true_data_file):
            df_true = pd.read_csv(true_data_file, index_col=0).T
            true_norm_data, _, _, _, _, _ = norm_and_eval_view(df_true, true_label, 'True')
        
        # Compute Metrics
        #_, _, _, _, _, _ = norm_and_eval_view(df_drop_data, true_label, 'Raw')
        imp_norm_data, ari_val, ji_val, nmi_val, ps_val, sw_val = norm_and_eval_view(df_Imp, true_label, 'GDCC')
        
        pcc_val, l1_val, rmse_val = 0.0, 0.0, 0.0
        if true_norm_data is not None:
            pcc_val = np.round(calculate_pearson(true_norm_data, imp_norm_data), 3)
            l1_val = np.round(calculate_l1(true_norm_data, imp_norm_data), 3)
            rmse_val = np.round(calculate_rmse(true_norm_data, imp_norm_data), 3)

        result_row = {
            'Dataset': dataset,
            'PCC': pcc_val,
            'L1': l1_val,
            'RMSE': rmse_val,
            'ARI': ari_val,
            'JI': ji_val,
            'NMI': nmi_val,
            'PS': ps_val,
            'ASW': sw_val
        }
        all_results.append(result_row)
        print(f"  Results -> PCC: {pcc_val}, ARI: {ari_val}, RMSE: {rmse_val}")

    # Save Summary
    if all_results:
        df_res = pd.DataFrame(all_results)
        summary_path = os.path.join(args.output_dir, 'GDCC_evaluation_summary.csv')
        df_res.to_csv(summary_path, index=False)
        print(f"\nAll results saved to: {summary_path}")
        print(df_res.to_string())

if __name__ == "__main__":
    main()