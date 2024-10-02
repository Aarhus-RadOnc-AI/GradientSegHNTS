import json
import random
from sklearn.model_selection import KFold


"""
To address the issue properly, we should first gather all the data and then reapply a random split 
while ensuring that the patientid and patientidpre are always in the same fold and that no pre data
is included in the validation set.
"""
# Load the existing splits JSON file
with open('/mnt/processing/jintao/nnUNet_preprocessed/Dataset504_midRT_geodist/splits_final_nnunet.json', 'r') as f:
    splits = json.load(f)

# Step 1: Collect all unique patient IDs without their 'pre' counterparts
patientid_list = set()

# Collect only 'patientid', not 'pre'
for fold in splits:
    for patient in fold['train'] + fold['val']:
        if not patient.endswith('pre'):
            patientid_list.add(patient)

patientid_list = list(patientid_list)  # Convert set back to list

# Step 2: Perform a 5-fold split on only the 'patientid'
kf = KFold(n_splits=5, shuffle=True, random_state=42)
new_splits = []

# Step 3: Create new folds by adding 'patientidpre' to train but excluding it from validation
for train_idx, val_idx in kf.split(patientid_list):
    train_patients = []
    val_patients = []

    # Add 'patientid' to train and val as per the split
    for idx in train_idx:
        patientid = patientid_list[idx]
        train_patients.append(patientid)

    for idx in val_idx:
        patientid = patientid_list[idx]
        val_patients.append(patientid)

    # Step 4: For each train fold, add corresponding 'patientidpre' (if not already added)
    for patientid in train_patients[:]:  # Create a shallow copy to avoid modification during iteration
        pre_patientid = f"{patientid}pre"
        if pre_patientid not in train_patients:
            train_patients.append(pre_patientid)  # Add 'pre' counterpart to training only

    new_splits.append({
        'train': train_patients,
        'val': val_patients
    })

# Step 5: Check for contamination and ensure no overlap between train and validation sets
for i, fold in enumerate(new_splits):
    train_patients_set = set([p.replace('pre', '') for p in fold['train']])
    val_patients_set = set(fold['val'])

    print(f"Fold {i+1}:")
    print(f"  Train Set: {len(fold['train'])} patients (including 'pre' data)")
    print(f"  Val Set: {len(fold['val'])} patients")
    print(f"  Train-Val Overlap: {train_patients_set.intersection(val_patients_set)}")
    assert len(train_patients_set.intersection(val_patients_set)) == 0, "Train and validation sets overlap!"

# Step 6: Save the new split file
with open('/mnt/processing/jintao/nnUNet_preprocessed/Dataset504_midRT_geodist/splits_final.json', 'w') as f:
    json.dump(new_splits, f, indent=4)
