#!/bin/bash

# Set the base directory where the folds are located
FOLDS_DIR="/data/jintao/nnUNet/nnUNet_results/Dataset504_midRT_geodist/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres_bs8"

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
        INPUT_FOLDER="${FOLDS_DIR}/${fold}/margins/${margin}/images"
        OUTPUT_FOLDER="${FOLDS_DIR}/${fold}/margins/${margin}/preds"

        # Create the output folder if it doesn't exist
        mkdir -p "${OUTPUT_FOLDER}"

        # Print a message indicating which fold and margin are being processed
        echo "Processing ${fold} with ${margin} margin..."

        # Run the nnUNet prediction
        CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict -i "${INPUT_FOLDER}" -o "${OUTPUT_FOLDER}" -d 504 -f "${fold_num}" -c 3d_fullres_bs8  -p nnUNetResEncUNetMPlans --disable_tta

        # Print a message when done
        echo "Prediction for ${fold} with ${margin} margin completed!"
    done
done

echo "All predictions completed!"