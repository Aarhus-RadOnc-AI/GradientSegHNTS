# GradientSegHNTS

# nnUNet Customize: Enhanced UMamba for Head and Neck Tumor Segmentation

## Overview
This project presents a customized version of nnUNet, optimized for head and neck tumor segmentation (HNTS). We enhance the original UMamba architecture by introducing significant optimizations to improve computational efficiency and segmentation accuracy.

## Key Modifications
- **Removal of the Mamba Layer and Residual Blocks**: We optimize UMamba by removing the Mamba layer in the first block and the residual blocks in the decoder, significantly enhancing computational efficiency while preserving the model's ability to capture long-range dependencies.
- **Integration with nnU-Net ResEnc**: By combining UMamba’s long-range dependency modeling with nnU-Net ResEnc’s enhanced residual encoding, we achieve improved accuracy in Gross Tumor Volume (GTV) delineation, especially in the complex anatomical structures of head and neck cancer.
