from setuptools import setup, find_packages

setup(
    name="scvedge",
    version="0.1.0",
    author="Xie Lingyun",
    author_email="lynnnx1208@gmail.com",
    description="A VAE/GAN framework for robust single-cell multi-omics integration",
    url="https://github.com/Lynn-nn-n/scVEDGE",  # GitHub 地址
    packages=['scvedge'],
    include_package_data=False,
    python_requires=">=3.8,<3.13",
    install_requires=[
        "torch>=2.0.0,<3.0.0",
        "lightning>=2.0.0,<3.0.0",
        "optax>=0.1.6",
        "numpy>=1.21,<2.0",
        "pandas>=1.3",
        "scipy>=1.7,<2.0",
        "anndata>=0.8,<1.0",
        "scvi-tools==1.2",
        "typing-extensions>=4.0,<5.0",
        "requests",
    ],## not sure bout this
)
