# scVEDGE

**scVEDGE** (VAE Enhanced by Divergence-aware GAN for integrative Embedding) is a deep generative framework for robust and flexible integration of single-cell multi-omics data. It supports cross-modal inference, handles unpaired and heterogeneous datasets, and produces biologically meaningful latent embeddings.

---

## 🌟 Features

- 🔄 **Multi-modal Integration** (RNA, ATAC, Protein)
- 🧬 **Cross-modality Generation**
- ⚙️ **Unsupervised Learning** (no labels needed)
- 🛠️ **Batch Correction**
- 🎯 **Biologically Consistent Embeddings**

---

## 🗂️ Project Structure

scVEDGE/  
├── Data/ # Input datasets, preprocessed data, and resultant data from experiments conducted in paper  
├── model_trained/ # Trained models from experiments conducted in paper  
├── scripts/ # Training / evaluation / inference scripts for experiments conducted in paper  
├── scripts_baseline/  # Scripts implementing baseline or comparison methods used in experiments to evaluate the performance of our proposed approach.  
├── scvedge/ # Core model code (modules, architectures)  
│ └── README.md  
│ └── setup.py  


---

## 🚀 Installation

```
#bash
git clone https://github.com/your-username/scVEDGE.git  #github地址
cd scVEDGE
pip install .
```
Make sure you have Python < 3.13 (We suggest 3.12).

---

## 📂 Input Format and usage
### Input Format
We organize our multimodal single-cell data using an AnnData object, which is capable of including three modalities: RNA expression, ATAC accessibility, and protein expression.  

The RNA and ATAC data are concatenated along the feature axis and stored in the .X matrix. The first set of columns corresponds to gene expression, and the remaining columns correspond to chromatin accessibility peaks. The .var["modality"] field is used to annotate each feature, labeling it as either "gene_expression" or "peaks", which allows for modality-specific operations on .X.  

The protein expression data is stored separately in .obsm["protein_expression"].  

To account for missing modalities, we fill the corresponding data entries (in .X or .obsm) with zeros, ensuring a uniform structure across all cells regardless of available modalities.  

On the cell level, .obs["modality"] is used to annotate the modalities available for each cell, such as "exp+acc" or "exp".  

Optionally, .obs["cell_type"] may be provided if cell-type annotations are available, but it is not required for model training.  
### Training
Say if you already have preprocessed anndata stored in .h5ad format:
```
adata = anndata.read("../Data/adata_raw.h5ad.gz")
```
This could be an example to train the corresponding model:
```
VEDGE.setup_anndata(adata, batch_key='modality', categorical_covariate_keys=['rep', 'tech'])
#batch_key: This parameter specifies the key in the adata.obs DataFrame that represents the 
#batch/modality information.
#categorical_covariate_keys: This parameter takes a list of keys from adata.obs that 
#represent categorical covariates. These covariates are additional metadata features (like 
#“replicate” or “technique”) that can influence the observed data.

torch.set_float32_matmul_precision('high')
mvi = VEDGE(adata, 
                        n_genes = (adata.var.modality=='Gene Expression').sum(), 
                        n_regions = (adata.var.modality=='Peaks').sum(),
                        fully_paired=False #if not fully-paired on RNA and ATAC
            )
mvi.train(batch_size = 32,
             early_stopping = True,
             early_stopping_patience = 50,
             adversarial_mixing=True)
mvi.save("../model_trained/mix_source_vedge")
```
### Evaluation
Below is an example of how to visualize the embedding distribution using UMAP.
```
#Getting latent representations
mvi = VEDGE.load("../model_trained/mix_source_vedge", adata)
adata.obsm["X_vedge"] = mvi.get_latent_representation()

#umap
sc.pp.neighbors(adata, use_rep="X_vedge")
sc.tl.umap(adata, min_dist=0.2)
adata.obsm["umap"] = umap.UMAP(random_state=420).fit_transform(adata.obsm["X_vedge"])
sns.scatterplot(
    x=adata.obsm["umap"][:,0],
    y=adata.obsm["umap"][:,1],
    hue=adata_plt.obs.modality, #colored by modality
    s=1,
    palette='bright',
)
plt.legend(title='', loc='lower left')
```
### Impute missing modalities
You can use:
```
adata_rna = adata[adata.obs.modality == 'expression']
atac_imputed = mvi.get_accessibility_estimates(adata_rna)
```
to get the corresponding imputed atac values of adata_rna, and
```
adata_atac = adata[adata.obs.modality == 'accessibility']
rna_imputed = mvi.get_normalized_expression(adata_atac)
```
to get rna_imputed.

---

## 📬 Contact
For questions, feel free to open an issue or contact lynnnx1208@gmail.com or 2022112509@stu.hit.edu.cn.

---

## 🤝 Acknowledgments
This project builds upon work in variational autoencoders (VAE), generative adversarial networks (GAN), and multi-omics integration frameworks MultiVI.

