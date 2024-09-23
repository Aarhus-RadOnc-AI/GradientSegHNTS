#!/bin/bash

for fold in {0..4}
do
  echo "Processing fold $fold"
  python ../evaluation.py /mnt/processing/jintao/nnUNet_preprocessed/Dataset504_midRT_geodist/gt_segmentations /data/jintao/nnUNet/nnUNet_results/Dataset504_midRT_geodist/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres_bs8/fold_$fold/validation
done
