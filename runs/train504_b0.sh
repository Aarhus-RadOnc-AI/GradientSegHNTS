#!/bin/bash
#nnUNetv2_plan_experiment -d 504 -pl nnUNetPlannerResEncM -gpu_memory_target 48 -overwrite_plans_name nnUNetResEncUNetPlans_48G
#CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 504 3d_fullres_bs8 0 -p nnUNetResEncUNetMPlans
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train --c 504 3d_fullres_bs8 2 -p nnUNetResEncUNetMPlans #750
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train --c 501 3d_fullres_bs4 1 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerUmamba #650

#CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 504 3d_fullres_bs8 0 -p nnUNetResEncUNetMPlans -tr nnUNetTraineResenc

