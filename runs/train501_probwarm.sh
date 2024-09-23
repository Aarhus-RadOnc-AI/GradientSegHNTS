#!/bin/bash
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainerProbSwinUMambaWarmup
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 501 3d_fullres 1 -tr nnUNetTrainerProbSwinUMambaWarmup

