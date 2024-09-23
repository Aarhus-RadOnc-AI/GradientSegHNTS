import os
import SimpleITK as sitk
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import warnings

# Set folder paths
imageTr_folder = "/data/jintao/nnUNet/nnUNet_raw_data_base/Dataset504_midRT_geodist/imagesTr"
labelTr_folder = "/data/jintao/nnUNet/nnUNet_raw_data_base/Dataset504_midRT_geodist/labelsTr"
imageTr_crop_folder = "/data/jintao/nnUNet/nnUNet_raw_data_base/Dataset504_midRT_geodist/imagesTr_crop"
labelTr_crop_folder = "/data/jintao/nnUNet/nnUNet_raw_data_base/Dataset504_midRT_geodist/labelsTr_crop"

# Create output directories if they don't exist
os.makedirs(imageTr_crop_folder, exist_ok=True)
os.makedirs(labelTr_crop_folder, exist_ok=True)


def crop_image_and_label(patient_id, margin=50):
    try:
        # File paths for both modalities
        image1_path = os.path.join(imageTr_folder, f"{patient_id}_0000.nii.gz")
        image2_path = os.path.join(imageTr_folder, f"{patient_id}_0001.nii.gz")  # New modality
        label_path = os.path.join(labelTr_folder, f"{patient_id}.nii.gz")

        # Output paths
        image1_out_path = os.path.join(imageTr_crop_folder, f"{patient_id}_0000.nii.gz")
        image2_out_path = os.path.join(imageTr_crop_folder, f"{patient_id}_0001.nii.gz")  # New modality
        label_out_path = os.path.join(labelTr_crop_folder, f"{patient_id}.nii.gz")
        
        # Read images
        image1 = sitk.ReadImage(image1_path)
        image2 = sitk.ReadImage(image2_path)  # New modality
        label = sitk.ReadImage(label_path)
        spacing = image1.GetSpacing()

        # Convert images to numpy arrays for manipulation
        image1_array = sitk.GetArrayFromImage(image1)
        image2_array = sitk.GetArrayFromImage(image2)  # New modality
        label_array = sitk.GetArrayFromImage(label)

        # Get image shape (z, y, x) -> axial, coronal, sagittal
        img_shape = image1_array.shape
    
        fixed_spacing = [1.199997067451477, 0.5, 0.5]

        # Determine cropping margins based on image shape
        if 550 < image1_array.shape[-1] < 600:
            margin = 50
        elif image1_array.shape[-1] >= 600:
            if spacing[0] <= fixed_spacing[-1]:
                margin = 110
            else:
                margin = 120
            if image1_array.shape[-1] >= 700:
                margin = 160
        else:
            margin = 30
            
        # Crop the images (keep all slices along the axial axis)
        cropped_image1_array = image1_array[:, margin:-margin, margin:-margin]
        cropped_image2_array = image2_array[:, margin:-margin, margin:-margin]  # New modality
        cropped_label_array = label_array[:, margin:-margin, margin:-margin]
        
        # Check if any label volume was removed
        original_label_volume = np.sum(label_array > 0)
        cropped_label_volume = np.sum(cropped_label_array > 0)
        
        if cropped_label_volume < original_label_volume:
            warnings.warn(f"@@Warning: Cropping may have removed some label volume for patient {patient_id}.")
        
        # Adjust origin after cropping (since margin is removed)
        original_origin = image1.GetOrigin()
        original_spacing = image1.GetSpacing()

        new_origin = (
            original_origin[0] + margin * original_spacing[0],
            original_origin[1] + margin * original_spacing[1],
            original_origin[2]
        )
        
        # Convert cropped numpy arrays back to SimpleITK images
        cropped_image1 = sitk.GetImageFromArray(cropped_image1_array)
        cropped_image2 = sitk.GetImageFromArray(cropped_image2_array)  # New modality
        cropped_label = sitk.GetImageFromArray(cropped_label_array)
        
        # Update metadata (spacing, direction, new origin)
        cropped_image1.SetSpacing(image1.GetSpacing())
        cropped_image1.SetDirection(image1.GetDirection())
        cropped_image1.SetOrigin(new_origin)
        
        cropped_image2.SetSpacing(image2.GetSpacing())  # New modality
        cropped_image2.SetDirection(image2.GetDirection())  # New modality
        cropped_image2.SetOrigin(new_origin)  # New modality
        
        cropped_label.SetSpacing(label.GetSpacing())
        cropped_label.SetDirection(label.GetDirection())
        cropped_label.SetOrigin(new_origin)
        
        # Write the cropped images and label
        sitk.WriteImage(cropped_image1, image1_out_path)
        sitk.WriteImage(cropped_image2, image2_out_path)  # New modality
        sitk.WriteImage(cropped_label, label_out_path)
        
        print(f"Successfully processed patient {patient_id}.")
        
        # Return necessary information for restoring the original image after prediction
        return {
            'patient_id': patient_id,
            'crop_margin': margin,
            'original_origin': original_origin,
            'original_spacing': original_spacing,
            'new_origin': new_origin,
        }
    except Exception as e:
        print(f"Error processing patient {patient_id}: {e}")
        return None

def main():
    # Get list of patient IDs based on label file names
    patient_ids = [f.split(".nii.gz")[0] for f in os.listdir(labelTr_folder) if f.endswith(".nii.gz")]

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        executor.map(crop_image_and_label, patient_ids)

if __name__ == "__main__":
    main()