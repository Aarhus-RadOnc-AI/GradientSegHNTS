#!/bin/bash

# Define the source directories (fold_0 to fold_4)
FOLD_DIRS=("fold_0" "fold_1" "fold_2" "fold_3" "fold_4")

# Define the target directory (validation)
TARGET_DIR="validation"

# Create the target directory if it does not exist
if [ ! -d "$TARGET_DIR" ]; then
  mkdir -p "$TARGET_DIR"
  echo "Created validation directory: $TARGET_DIR"
fi

# Loop through each fold directory
for FOLD in "${FOLD_DIRS[@]}"; do
  # Check if the fold directory exists
  if [ -d "$FOLD" ]; then
    echo "Copying files from $FOLD to $TARGET_DIR..."
    
    # Copy all .nii.gz files from the current fold directory to the target directory
    cp "$FOLD"/*.nii.gz "$TARGET_DIR"/
    
    echo "Finished copying files from $FOLD."
  else
    echo "Directory $FOLD does not exist. Skipping..."
  fi
done