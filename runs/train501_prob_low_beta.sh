#!/bin/bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train --c 501 3d_fullres 0 -tr nnUNetTrainerProbSwinUMamba_low_beta

