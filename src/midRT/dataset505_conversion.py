import os
import numpy as np
import SimpleITK as sitk
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from scipy.ndimage import label
from nnunetv2.paths import nnUNet_raw
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
base_dir = '/data/jintao/nnUNet/HNTSMRG24_train/'
dataset_preRT = 'Dataset501_preRT'
dataset_midRT = 'Dataset502_midRT'
dataset_grad505 = 'Dataset505_grad_map_filtered'
preRT_label_path = os.path.join(nnUNet_raw, dataset_preRT, 'labelsTr')
midRT_label_path = os.path.join(nnUNet_raw, dataset_midRT, 'labelsTr')

# Function to calculate volume differences for a single patient file
def analyze_patient_volume(patient_file, value):
    preRT_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(preRT_label_path, patient_file)))
    midRT_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(midRT_label_path, patient_file)))
    
    preRT_labeled, num_features_preRT = label(preRT_mask == value)
    patient_data = []

    for i in range(1, num_features_preRT + 1):
        preRT_volume = np.sum(preRT_labeled == i)
        coords = np.argwhere(preRT_labeled == i)
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        bbox_slice = tuple(slice(min_c, max_c + 1) for min_c, max_c in zip(min_coords, max_coords))

        midRT_in_bbox = midRT_mask[bbox_slice]
        midRT_instance_present = np.any(midRT_in_bbox == value)
        cured_status = 1 if midRT_instance_present else 0

        patient_data.append((preRT_volume, cured_status))

    return patient_data

# Function to analyze volume differences using multithreading
def analyze_volume_differences(preRT_label_path, midRT_label_path, value):
    data = []
    patient_files = [f for f in os.listdir(preRT_label_path) if f.endswith('.nii.gz')]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(analyze_patient_volume, patient_file, value): patient_file for patient_file in patient_files}

        for future in as_completed(futures):
            try:
                patient_data = future.result()
                data.extend(patient_data)
            except Exception as exc:
                patient_file = futures[future]
                logging.error(f"Patient file {patient_file} generated an exception: {exc}")

    return np.array(data)

# Separate analysis for GTVp (value=1) and GTVn (value=2)
def run_analysis(value, label_name):
    data = analyze_volume_differences(preRT_label_path, midRT_label_path, value)
    preRT_volumes = data[:, 0].reshape(-1, 1)
    cured_status = data[:, 1]

    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(preRT_volumes, cured_status, test_size=0.2, random_state=42)

    # Scale the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use SMOTE to balance the training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    # Train Random Forest with cross-validation on training data
    rf_model = RandomForestClassifier(random_state=42)
    cv_scores = cross_val_score(rf_model, X_train_smote, y_train_smote, cv=5, scoring='roc_auc')
    rf_model.fit(X_train_smote, y_train_smote)

    # Evaluate on test data
    y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    y_pred = rf_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    # Log results
    logging.info(f"{label_name} Cross-Validation ROC-AUC Scores: {cv_scores}")
    logging.info(f"{label_name} Mean CV ROC-AUC: {np.mean(cv_scores)}")
    logging.info(f"{label_name} Test ROC-AUC Score: {roc_auc}")
    logging.info(f"{label_name} Test Accuracy Score: {accuracy}")
    logging.info(f"{label_name} Classification Report on Test Data:\n{classification_report(y_test, y_pred)}")

    # Calculate and log the optimal threshold based on recall = 1
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    # Find the threshold for the point where TPR (recall) is 1
    recall_1_idx = np.where(tpr == 1)[0]
    if len(recall_1_idx) > 0:
        optimal_idx = recall_1_idx[-1]  # Choose the last point where recall is 1 to maximize FPR
        optimal_threshold = thresholds[optimal_idx]
        logging.info(f"{label_name} Optimal threshold with recall=1 and highest FPR: {optimal_threshold}")
    else:
        logging.warning(f"{label_name} No threshold found with recall=1.")
        optimal_threshold = None

    return optimal_threshold, y_test, y_pred_proba

# Run analysis for GTVp (value=1) and GTVn (value=2)
gtvp_threshold, y_test_gtvp, y_pred_proba_gtvp = run_analysis(1, 'GTVp')
gtvn_threshold, y_test_gtvn, y_pred_proba_gtvn = run_analysis(2, 'GTVn')

# Create a figure with side-by-side subplots
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Plot ROC curve for GTVp
fpr_gtvp, tpr_gtvp, thresholds_gtvp = roc_curve(y_test_gtvp, y_pred_proba_gtvp)
roc_auc_gtvp = auc(fpr_gtvp, tpr_gtvp)

ax[0].plot(fpr_gtvp, tpr_gtvp, color='darkorange', lw=2, label=f'GTVp ROC curve (area = {roc_auc_gtvp:.2f})')
ax[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax[0].set_xlim([0.0, 1.0])
ax[0].set_ylim([0.0, 1.05])
ax[0].set_xlabel('False Positive Rate')
ax[0].set_ylabel('True Positive Rate')
ax[0].set_title('Receiver Operating Characteristic (ROC) Curve for GTVp')
ax[0].legend(loc='lower right')

# Plot ROC curve for GTVn
fpr_gtvn, tpr_gtvn, thresholds_gtvn = roc_curve(y_test_gtvn, y_pred_proba_gtvn)
roc_auc_gtvn = auc(fpr_gtvn, tpr_gtvn)

ax[1].plot(fpr_gtvn, tpr_gtvn, color='blue', lw=2, label=f'GTVn ROC curve (area = {roc_auc_gtvn:.2f})')
ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax[1].set_xlim([0.0, 1.0])
ax[1].set_ylim([0.0, 1.05])
ax[1].set_xlabel('False Positive Rate')
ax[1].set_ylabel('True Positive Rate')
ax[1].set_title('Receiver Operating Characteristic (ROC) Curve for GTVn')
ax[1].legend(loc='lower right')

# Save combined figure as SVG file
plt.tight_layout()
plt.savefig('roc_curves_gtvp_gtvn.svg', format='svg')
plt.show()
