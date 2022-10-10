# scRare

## Introduction

scBalance, a neural network frame work for novel rare cell deteciton, provides a fast, accurate and user-friendly novel rare cell detection for a new single-cell RNA-seq profile. By leveraging the newly designed neural network structure, scBalance especially obtains an outperformance on rare cell type annotation and robustness on batch effect. 

Notably, scBalance is the only tool that is highly compatible with Scanpy. Users can easily use it with the Anndata structure during the analysis. Instructions and examples are provided in the following tutorials.

## Requirement

- Scanpy (Compatible with all versions)
- Pytorch (With Cudatoolkit is recommanded)
- Numpy > 1.20
- Pandas > 1.2

## Installation

```Python
git clone https://github.com/chuanyang-Zheng/scRare.git
```

## Tutorial

- [scRare Tutorial](https://github.com/yuqcheng/scBalance/blob/main/Tutorial/scBalance%20Tuotrial_Annotation%20of%203k%20PBMCs.ipynb)

[//]: # (## Usage)

[//]: # ()
[//]: # (### 0. Data preprocessing)

[//]: # ()
[//]: # (We design a ```Scanpy_Obj_IO``` module for users to preprocess the input data to the input format of the scBalance. The use of this module can be seen in the Tutorial [Annotation of 3k PBMCs]&#40;https://github.com/yuqcheng/scBalance/blob/main/Tutorial/scBalance%20Tuotrial_Annotation%20of%203k%20PBMCs.ipynb&#41;.)

[//]: # ()
[//]: # (```Python)

[//]: # (import scRare.scRare_IO as ss)

[//]: # (ss.Scanpy_Obj_IO&#40;test_obj=adata, ref_obj=train_adata, label_obj=train_label, scale = False&#41;)

[//]: # (```)

[//]: # ()
[//]: # (For users who want to process by yourselve, please follow the Tutorial [scRarE Tutorial]&#40;https://github.com/yuqcheng/scBalance/blob/main/Tutorial/scBalance%20Tuotrial_Annotation%20of%203k%20PBMCs.ipynb&#41;.)

[//]: # ()
[//]: # (### 1. The inputs of scBalance are two expression matrices and one label vector. )

[//]: # ()
[//]: # (```Python)

[//]: # (import scRare as sb)

[//]: # (pred_result = sb.Rare&#40;test, reference, label, processing_unit&#41;)

[//]: # (```)

[//]: # ()
[//]: # (in which )

[//]: # ()
[//]: # (- **test=The expression matrix of the sample to be annotated**,)

[//]: # (- **reference=The expression matrix of the labeled dataset &#40;reference set&#41;,** )

[//]: # (- **label = label vector &#40;in pandas structure&#41;**,)

[//]: # (- **processing_unit = 'cpu'&#40;Default&#41;/'gpu'**. If no changes, the default processor will be CPU. We highly recommend setting as 'gpu' if your server supports.)

[//]: # ()
[//]: # (Column name can be anything.)

[//]: # ()
[//]: # (### 2. Waiting for the progress bar to finish.)

[//]: # ()
[//]: # (```)

[//]: # (--------Start annotating----------)

[//]: # (Computational unit be used is: cuda)

[//]: # (100%[====================->]28.94s)

[//]: # (--------Annotation Finished----------)
```

## Citation
scRare: a neural netowrk framework for novel rare cell detection of single-cell transcriptome data. Chuanyang Zheng, Yuqi Cheng, Xuesong Wang, Yixuan, Wang, Yu Li.
