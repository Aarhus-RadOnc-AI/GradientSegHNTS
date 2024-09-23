#!/bin/bash


CUDA_VISIBLE_DEVICES=1 nnUNetv2_train --val 501 3d_fullres 0 -tr nnUNetTrainerProbSwinUMamba
