import scanpy as sc
import mofapy2
from mudata import MuData
import anndata
from mofapy2.run.entry_point import entry_point
import numpy as np
import pandas as pd
from scipy.sparse import issparse
import os
import torch

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1)

adata=anndata.read("../Data/my_dogma_all_genes_cells_dig_ctrl_75unpaired.h5ad.gz") 
# 拆分 RNA 和 ATAC 模态
adata_rna = adata[:,adata.var['Type'] == 'Gene Expression'].copy()
mask = adata_rna.obs['modality'] == 'acc+pro'
# 如果是稀疏矩阵，先转成 dense（否则不能直接用 np.nan）
if issparse(adata_rna.X):
    adata_rna.X = adata_rna.X.toarray()
# 将这部分细胞的表达设为 NaN
adata_rna.X[mask.values, :] = np.nan
adata_atac = adata[:, adata.var['Type'] == 'Peaks'].copy()
mask = adata_atac.obs['modality'] == 'exp+pro'
# 如果是稀疏矩阵，先转成 dense（否则不能直接用 np.nan）
if issparse(adata_atac.X):
    adata_atac.X = adata_atac.X.toarray()
# 将这部分细胞的表达设为 NaN
adata_atac.X[mask.values, :] = np.nan

# 蛋白质模态
adata_protein = sc.AnnData(adata.obsm['protein_expression'])
#adata_protein.var_names = adata.uns['protein_expression_columns']  # 假设蛋白名称已存储
adata_protein.var_names = [f"Protein_{i}" for i in range(adata_protein.n_vars)]
adata_protein.obs_names = adata.obs_names.astype(str)  # 确保细胞ID一致
adata_protein.obs = adata.obs.copy()  # 复制细胞信息
adata_protein.X[adata_protein.obs.modality == 'exp+acc', :] = np.nan

adata_atac.var_names_make_unique()
adata_rna.var_names_make_unique()

data = [[adata_rna.X], [adata_atac.X], [adata_protein.X]] 
views_names = ["rna", "atac", "protein"]
features_names = [
    adata_rna.var_names.tolist(),
    adata_atac.var_names.tolist(),
    adata_protein.var_names.tolist()
]
samples_names = [adata_rna.obs_names.tolist()]  # 假设所有视图的 obs 是一致的
samples_groups = adata_rna.obs["sample_type"].astype(str).tolist()  # 可替换为你想用的列
#groups_names = sorted(set(samples_groups))

# 初始化模型
print(adata.obs['sample_type'].unique())
ent = entry_point()
ent.set_data_options(scale_groups=True, scale_views=True, use_float32=True)  # 自动标准化
print('start set_data_df')
ent.set_data_matrix(
    data=data,
    views_names=views_names,
    features_names=features_names,
    samples_names=samples_names,
    #samples_groups=samples_groups,   # 可选
    #groups_names=samples_groups       # 可选
)
print('end set_data_df')

# 设置模型参数
ent.set_model_options(
    factors=20,              # 根据数据量调整（建议20-50）
    spikeslab_weights=True,  # 启用稀疏权重
    ard_factors=True        # 自动相关性确定
)
ent.set_train_options(     
    dropR2=0.01,           # 提前停止阈值
    seed=42                # 随机种子
)

# 训练模型
ent.build()  # 构建模型
ent.run()

#ent.save("../model_trained/mofa_pro/mofa_model.hdf5",save_parameters=True)  # 保存模型参数

# 提取因子嵌入（cells × factors）
factors = ent.model.nodes["Z"].getExpectation()
adata.obsm["X_mofa"] = factors # 保存到原 adata

adata.write("../Data/dogma_all_genes_cells_75unpaired_processed_mofa.h5ad.gz", compression="gzip")
print("Saved processed data to disk.")
# 保存模型参数（HDF5格式）
ent.save("../model_trained/mofa_pro/mofa_model2.hdf5")

# 重新加载模型
#ent_loaded = entry_point()
#ent_loaded.load("mofa_model.hdf5")