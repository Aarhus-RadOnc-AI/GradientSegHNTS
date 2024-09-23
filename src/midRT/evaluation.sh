#!/bin/bash

# Set the base directory where the folds are located
FOLDS_DIR="/data/jintao/nnUNet/nnUNet_results/Dataset504_midRT_geodist/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres_bs8"
GT_DIR="/data/jintao/nnUNet/nnUNet_raw_data_base/Dataset504_midRT_geodist/labelsTr"
# List of folds
folds=("fold_0" "fold_1" "fold_2" "fold_3" "fold_4")

# List of voxel margins
margins=("2voxels" "4voxels" "6voxels" "8voxels")

# Iterate over each fold
for fold in "${folds[@]}"; do
    # Extract the integer part of the fold (e.g., 0, 1, 2, 3, 4)
    fold_num="${fold##*_}"

    # Iterate over each margin folder
    for margin in "${margins[@]}"; do
        # Define the input and output directories
        OUTPUT_FOLDER="${FOLDS_DIR}/${fold}/margins/${margin}/preds"

        # Print a message indicating which fold and margin are being processed
        echo "evaluating ${fold} with ${margin} margin..."

        # Run the Evaluation
        python evaluation.py "${GT_DIR}" "${OUTPUT_FOLDER}" 

        # Print a message when done
        echo "Evaluation for ${fold} with ${margin} margin completed!"
    done
done

echo "All Evaluation completed!"