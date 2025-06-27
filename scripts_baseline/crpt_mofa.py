import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from mofapy2.run.entry_point import entry_point
import scipy.io
import umap
from scipy.spatial.distance import cdist
import anndata
import scanpy as sc
#import scvi
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import issparse
#from egd_model import MULTIVI #as X
#from cobolt.model import Cobolt

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
'''FIGPATH='../Figures/my_crpt'
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
torch.set_num_threads(1)


def compute_batch_mixing(latent, batch_id, k):
    nng = kneighbors_graph(latent, n_neighbors=k).tocoo()
    batch_id = pd.Categorical(batch_id).codes
    self_id = batch_id[nng.row]
    ne_id = batch_id[nng.col]
    
    _, c = np.unique(batch_id, return_counts=True)
    theoretic_score = ((c / c.sum()) ** 2).sum()
    return (self_id == ne_id).mean() / theoretic_score

def compute_foscttm(latent_acc, latent_exp):
    """
    计算 FOSCTTM 分数
    
    返回: 平均 FOSCTTM 值
    """
    distances = cdist(latent_acc, latent_exp, metric='euclidean')  # 计算所有细胞的欧式距离
    foscttm_values = []
    num_cells=latent_exp.shape[0]

    for i in range(num_cells):  # 遍历所有已知匹配的细胞对
        d_true = distances[i, i]  # 真实匹配的距离
        d_others = distances[i, :]  # 该细胞到所有其他细胞的距离
        
        rank = np.sum(d_others < d_true) / num_cells  # 计算 FOSCTTM
        foscttm_values.append(rank)

    return np.mean(foscttm_values)  # 返回平均 FOSCTTM 值

distances = []
enrichments = []
foscttms=[]
asw_modality=[]
NPPs = [0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1]
for npp in NPPs[1:]:
    print(npp)
    adata = anndata.read("../Data/my_corruption2/adata_r{}.h5ad.gz".format(int(npp * 100)))
    latent = adata.obsm['X_MultiVI']
    for K in [15,50,150,500]:
        print(K)
        enrichments.append((
            npp, 
            K, 
            compute_batch_mixing(latent, adata.obs.modality, K),
        ))
    latent_exp = latent[adata.obs.modality == 'expression']
    latent_acc = latent[adata.obs.modality == 'accessibility']
    distances.append(pd.DataFrame({
        'rate':npp,
        'distances':(((latent_exp - latent_acc) ** 2).sum(axis=1) ** 0.5),
    }))
    foscttms.append((npp,compute_foscttm(latent_acc,latent_exp)))
    asw_modality.append((npp,silhouette_score(latent, adata.obs["modality"], metric='euclidean')))

df_enrich = pd.DataFrame(enrichments, columns=('rate', 'K', 'enrichment'))
sns.scatterplot(data=df_enrich, x='rate', y='enrichment', hue='K')
plt.savefig(os.path.join(FIGPATH,'enrichment.png'), dpi=300, bbox_inches='tight')  # 保存为高分辨率PNG文件

dist_df = pd.concat(distances)
sns.violinplot(data=dist_df, x='rate', y='distances')
plt.savefig(os.path.join(FIGPATH,'dist.png'), dpi=300, bbox_inches='tight')  # 保存为高分辨率PNG文件

print('asw:',asw_modality)
print('foscttms:',foscttms)'''

adata = anndata.read("../Data/adata_raw.h5ad.gz")

def depair_anndata(adata, n_unpaired):
    modality_switch = np.where(adata.var.modality == "Peaks")[0].min()
    
    indices = np.arange(adata.shape[0])
    np.random.shuffle(indices)
    unpaired_idx = indices[:n_unpaired]
    paired_idx = indices[n_unpaired:]
    
    adata_p = adata[paired_idx].copy()
    adata_np_acc = adata[unpaired_idx].copy()
    adata_np_exp = adata_np_acc.copy()
    
    adata_np_acc.X[:, :modality_switch] = 0
    adata_np_exp.X[:, modality_switch:] = 0
    
    adata = anndata.AnnData(
        scipy.sparse.vstack((adata_p.X, adata_np_acc.X, adata_np_exp.X)),
        obs=pd.concat((adata_p.obs, adata_np_acc.obs, adata_np_exp.obs)),
        var=adata_p.var
    )
    adata.X.eliminate_zeros()
    
    has_chr = np.asarray(adata.X[:, modality_switch:].sum(axis=1) > 0).squeeze()
    has_rna = np.asarray(adata.X[:, :modality_switch].sum(axis=1) > 0).squeeze()
    adata.obs["modality"] = "expression"
    adata.obs.modality.loc[has_chr] = "accessibility"
    adata.obs.modality.loc[np.logical_and(has_chr, has_rna)] = "paired"
    return adata.copy()

def corrupt_and_process(adata, unpaired_rate):
    if os.path.exists("../Data/mofa_corruption/adata_r{}.h5ad.gz".format(int(unpaired_rate * 100))):
        print("already done, skipping!")
        return
    print("corrupting AnnData...")
    adata = depair_anndata(adata, int(unpaired_rate * adata.shape[0]))
    adata_rna = adata[:,adata.var['modality'] == 'Gene Expression'].copy()
    mask = adata_rna.obs['modality'] == 'accessibility'
# 如果是稀疏矩阵，先转成 dense（否则不能直接用 np.nan）
    if issparse(adata_rna.X):
        adata_rna.X = adata_rna.X.toarray()
# 将这部分细胞的表达设为 NaN
    adata_rna.X = adata_rna.X.astype(np.float32)
    adata_rna.X[mask.values, :] = np.nan
    adata_atac = adata[:, adata.var['modality'] == 'Peaks'].copy()
    mask = adata_atac.obs['modality'] == 'expression'
# 如果是稀疏矩阵，先转成 dense（否则不能直接用 np.nan）
    if issparse(adata_atac.X):
        adata_atac.X = adata_atac.X.toarray()
# 将这部分细胞的表达设为 NaN
    adata_atac.X = adata_atac.X.astype(np.float32)
    adata_atac.X[mask.values, :] = np.nan

    adata_atac.var_names_make_unique()
    adata_rna.var_names_make_unique()


    data = [[adata_rna.X], [adata_atac.X]] 
    views_names = ["rna", "atac"]
    features_names = [
      adata_rna.var_names.tolist(),
      adata_atac.var_names.tolist(),
    ]
    samples_names = [adata_rna.obs_names.tolist()]  # 假设所有视图的 obs 是一致的

    if len(samples_names[0]) == set(samples_names[0]):
        print("samples_names are unique")
    else:
        print("samples_names are not unique, please check!")
        samples_names[0] = list(range(len(samples_names[0])))

    print("training model...")
    ent = entry_point()
    ent.set_data_options(scale_groups=True, scale_views=True, use_float32=True)  # 自动标准化
    ent.set_data_matrix(
    data=data,
    views_names=views_names,
    features_names=features_names,
    samples_names=samples_names,
    #samples_groups=samples_groups,   # 可选
    #groups_names=samples_groups       # 可选
    )

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

# 提取因子嵌入（cells × factors）
    factors = ent.model.nodes["Z"].getExpectation()
    print("factors shape:", factors.shape)
    adata.obsm["X_mofa"] = factors

    sc.pp.neighbors(adata, use_rep="X_mofa")
    sc.tl.umap(adata, min_dist=0.2)
    adata.write("../Data/mofa_corruption/adata_r{}.h5ad.gz".format(int(unpaired_rate * 100)), compression='gzip')


NPPs = [0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1]
for npp in NPPs:
    print(npp)
    corrupt_and_process(adata, npp)