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
Make sure you have Python < 3.13

---

## 📂 Input Format and usage
See tutorial for more imformation.

---

## 📬 Contact
For questions, feel free to open an issue or contact lynnnx1208@gmail.com or 2022112509@stu.hit.edu.cn.

---

## 🤝 Acknowledgments
This project builds upon work in variational autoencoders (VAE), generative adversarial networks (GAN), and multi-omics integration frameworks MultiVI.

