#!/bin/bash
#CUDA_VISIBLE_DEVICES=2 nnUNetv2_train --val 501 3d_fullres 0 -tr nnUNetTrainerSwinUMamba
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train --c 501 3d_fullres 1 -tr nnUNetTrainerProbSwinUMamba_low_beta
#CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 501 3d_fullres 2 -tr nnUNetTrainerProbSwinUMamba_low_beta


