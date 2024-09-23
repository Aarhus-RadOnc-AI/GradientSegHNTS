#!/bin/bash
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train --c 501 3d_fullres 3 -tr nnUNetTrainerProbSwinUMamba_low_beta
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 501 3d_fullres 4 -tr nnUNetTrainerProbSwinUMamba_low_beta

