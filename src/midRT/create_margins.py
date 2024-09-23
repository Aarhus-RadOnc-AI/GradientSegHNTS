import os
import shutil
import numpy as np
import SimpleITK as sitk
import logging
from scipy.ndimage import distance_transform_edt
import scipy.ndimage as ndimage
from nnunetv2.paths import nnUNet_raw
from data_conversion import calculate_gradient_map_within_bbox

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
dataset_geodist = 'Dataset504_midRT_geodist'
folds_dir = '/data/jintao/nnUNet/nnUNet_results/Dataset504_midRT_geodist/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres_bs8'  # Adjust to where fold_0 to fold_4 directories are
dataset_dir504 = os.path.join(nnUNet_raw, dataset_geodist, 'imagesTr')# Adjust to where the full dataset is (with patientid_0000.nii.gz)
base_dir = '/data/jintao/nnUNet/HNTSMRG24_train/'


# Create margins for each fold
margins = [2, 4, 6, 8]
folds = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']

# Function to copy the T2 image
def copy_t2_image(patient_id, fold, margin):
    src = os.path.join(dataset_dir504, f"{patient_id}_0000.nii.gz")
    dest_dir = os.path.join(folds_dir, fold, f"margins/{margin}voxels/images/")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"{patient_id}_0000.nii.gz")
    shutil.copy(src, dest)

# Function to generate a gradient map with a fixed margin
def create_fixed_margin_gradient_map(patient_id, fold, margin):
    midRT_dir = os.path.join(base_dir, patient_id, 'midRT')
    
    preRT_mask_path = os.path.join(midRT_dir, f'{patient_id}_preRT_mask_registered.nii.gz')
    midRT_mask_path = os.path.join(midRT_dir, f'{patient_id}_midRT_mask.nii.gz')
    t2_image_path = os.path.join(dataset_dir504, f"{patient_id}_0000.nii.gz")
    
    preRT_mask_img = sitk.ReadImage(preRT_mask_path)
    preRT_mask = sitk.GetArrayFromImage(preRT_mask_img)
    
    midRT_mask_img = sitk.ReadImage(midRT_mask_path)
    midRT_mask_data = sitk.GetArrayFromImage(midRT_mask_img)
    
    t2_image_img = sitk.ReadImage(t2_image_path)
    t2_image = sitk.GetArrayFromImage(t2_image_img)

    gradient_map = calculate_gradient_map_within_bbox(t2_image, preRT_mask, midRT_mask_data, patient_id)
    
    # Save the gradient map
    gradient_img = sitk.GetImageFromArray(gradient_map.astype(np.float32))
    gradient_img.CopyInformation(t2_image_img)
    
    dest_dir = os.path.join(folds_dir, fold, f"margins/{margin}voxels/images/")
    os.makedirs(dest_dir, exist_ok=True)
    gradient_path = os.path.join(dest_dir, f"{patient_id}_0001.nii.gz")
    sitk.WriteImage(gradient_img, gradient_path)

# Function to process each fold
def process_fold(fold):
    validation_dir = os.path.join(folds_dir, fold, 'validation')
    
    # Get the patient IDs from the validation directory
    patients = [f.replace('.nii.gz', '') for f in os.listdir(validation_dir) if f.endswith('.nii.gz')]
    
    for patient_id in patients:
        logging.info(f"Processing patient {patient_id} for fold {fold}")
        
        # For each margin, copy the T2 image and create the gradient map
        for margin in margins:
            copy_t2_image(patient_id, fold, margin)
            create_fixed_margin_gradient_map(patient_id, fold, margin)

# Process each fold
for fold in folds:
    process_fold(fold)

logging.info("Finished processing all folds.")