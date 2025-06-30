import anndata, scvi, os, scanpy as sc, umap, matplotlib.pyplot as plt, seaborn as sns
import torch
from scvedge.vedge_model import VEDGE

torch.cuda.is_available()
torch.cuda.get_device_name(0)

def testing(mvi, save_path, pdf_path):
    mvi.train(adversarial_mixing=True)
    mvi.save(save_path)
    adata.obsm["X_multiVI"] = mvi.get_latent_representation()
    
    adata.obsm["umap_MultiVI"] = umap.UMAP(random_state=420).fit_transform(adata.obsm["X_multiVI"])
    
    sc.pp.neighbors(adata, use_rep="X_multiVI", key_added="multiVI")
    sc.tl.leiden(adata, neighbors_key="multiVI", key_added="multiVI_clusters", resolution=0.1)
    
    adata.write(pdf_path + "_processed.h5ad.gz", compression="gzip")


# ######################################################################################################################
adata = anndata.read("../Data/dogma_all_genes_cells_2ineach.h5ad.gz")
VEDGE.setup_anndata(adata, batch_key='modality' , protein_expression_obsm_key='protein_expression', categorical_covariate_keys=['sample_type'])
torch.set_float32_matmul_precision('high')
#adata.var['modality'] = adata.var['Type'].copy()
n_genes = (adata.var.modality=='Gene Expression').sum()
n_regions = (adata.var.modality=='Peaks').sum()
n_proteins = adata.obsm['protein_expression'].shape[1]

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

mvi = VEDGE(adata, n_genes=n_genes, n_regions=n_regions)
mvi.view_anndata_setup()
testing(mvi, save_path="../model_trained/Test3Mod_2ineach", pdf_path="../Data/dogma_all_genes_cells_2ineach")