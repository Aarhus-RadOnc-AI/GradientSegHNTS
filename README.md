# Gradient Map-Assisted Head and Neck Tumor Segmentation: A Pre-RT to Mid-RT Approach in MRI-Guided Radiotherapy (GradientSegHNTS)
This repository contains the code for two tasks from the HNTS-MRG 2024 challenge, focusing on the segmentation of Gross Tumor Volume in the primary (GTVp) and nodal (GTVn) regions for head and neck cancer. The tasks involve the segmentation of tumors in both pre-radiotherapy (pre-RT) and mid-radiotherapy (mid-RT) phases using advanced deep learning models.

## Task 1: Pre-RT Segmentation
In Task 1, we developed a model for segmenting pre-RT GTVp and GTVn using a combination of nnU-Net ResEnc M planner with UMamba. Below are some key contributions of this task:

- **Optimization of UMamba**: We removed the Mamba layer in the first block and the residual blocks in the decoder, significantly enhancing computational efficiency while maintaining the ability to capture long-range dependencies.
- **Improved Residual encoder**: By combining UMamba’s long-range dependency modeling with nnU-Net ResEnc’s enhanced residual encoding, we improved the accuracy of GTV delineation in the complex anatomy of head and neck cancer.

## Task 2: Mid-RT Segmentation
For Task 2, we tackled the challenge of mid-RT segmentation. Here are our key contributions:

- **Novel Approach Using Pre-RT Data**: We leveraged pre-RT tumor delineations to enhance mid-RT segmentation by identifying Regions of Interest (ROIs) around the tumors using deformably registered pre-RT data.
- **Gradient Maps for Improved Segmentation**: We computed gradient maps from the mid-RT T2w images and used them as additional input channels. Furthermore, we generated similar gradient maps from pre-RT images and their ground truth (GT) delineations to enrich the training data, effectively enhancing segmentation accuracy during the mid-RT phase.

This repository provides a comprehensive solution to the HNTS-MRG 2024 challenge, integrating advanced techniques for accurate head and neck tumor segmentation across different treatment phases.


## nnUNet Customize: Enhanced UMamba for Head and Neck Tumor Segmentation
Folder ```nnUNet``` presents a customized version of nnUNet, optimized for head and neck tumor segmentation (HNTS). We enhance the original UMamba architecture by introducing significant optimizations to improve computational efficiency and segmentation accuracy.

### Key Modifications
- **Removal of the Mamba Layer and Residual Blocks**: We optimize UMamba by removing the Mamba layer in the first block and the residual blocks in the decoder, significantly enhancing computational efficiency while preserving the model's ability to capture long-range dependencies.
- **Integration with nnU-Net ResEnc**: By combining UMamba’s long-range dependency modeling with nnU-Net ResEnc’s enhanced residual encoding, we achieve improved accuracy in Gross Tumor Volume (GTV) delineation, especially in the complex anatomical structures of head and neck cancer.
