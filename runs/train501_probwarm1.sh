#!/bin/bash
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 501 3d_fullres 2 -tr nnUNetTrainerProbSwinUMambaWarmup
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 501 3d_fullres 3 -tr nnUNetTrainerProbSwinUMambaWarmup
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 501 3d_fullres 4 -tr nnUNetTrainerProbSwinUMambaWarmup

