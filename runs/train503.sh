#!/bin/bash
#nnUNetv2_plan_experiment -d 504 -pl nnUNetPlannerResEncM -gpu_memory_target 48 -overwrite_plans_name nnUNetResEncUNetPlans_48G
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 504 3d_fullres_bs8 0 -p nnUNetResEncUNetMPlans
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 503 3d_fullres_bs8 0 -p nnUNetResEncUNetMPlans #750
