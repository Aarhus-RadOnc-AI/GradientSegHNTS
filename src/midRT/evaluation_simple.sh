#!/bin/bash

# Set the base directory where the folds are located
#PRED_DIR="/data/jintao/nnUNet/nnUNet_results/Dataset504_midRT_geodist/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres_bs8/fold_0/margins/4voxels/docker_new"
#PRED_DIR="/data/jintao/nnUNet/nnUNet_results/Dataset504_midRT_geodist/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres_bs8/fold_0/margins/4voxels/docker_preds"
PRED_DIR="/data/jintao/nnUNet/nnUNet_results/Dataset505_t2only/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres_bs8/fold_0/validation"
GT_DIR="/data/jintao/nnUNet/nnUNet_raw_data_base/Dataset504_midRT_geodist/labelsTr"

# Run the Evaluation
python evaluation.py "${GT_DIR}" "${PRED_DIR}" 
