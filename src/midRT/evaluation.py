import os
import SimpleITK as sitk
import numpy as np
import json
import concurrent.futures
import argparse
from medpy.metric.binary import hd95, assd
from scipy.stats import ttest_ind

# Function to compute volumes
def compute_volumes(im):
    spacing = im.GetSpacing()
    voxvol = spacing[0] * spacing[1] * spacing[2]
    stats = sitk.LabelStatisticsImageFilter()
    stats.Execute(im, im)
    nvoxels1 = stats.GetCount(1)
    nvoxels2 = stats.GetCount(2)
    return nvoxels1 * voxvol, nvoxels2 * voxvol

# Compute HD95 and MSD for both labels with voxel spacing
def compute_surface_distances(groundtruth, prediction, voxel_spacing):
    # Create binary masks for label 1 (GTVp) and label 2 (GTVn)
    groundtruth1 = sitk.GetArrayFromImage(groundtruth) == 1  # Binary mask for GTVp (label 1)
    groundtruth2 = sitk.GetArrayFromImage(groundtruth) == 2  # Binary mask for GTVn (label 2)
    prediction1 = sitk.GetArrayFromImage(prediction) == 1    # Binary mask for GTVp (label 1)
    prediction2 = sitk.GetArrayFromImage(prediction) == 2    # Binary mask for GTVn (label 2)

    # Convert boolean arrays to int (0 and 1)
    groundtruth1 = groundtruth1.astype(np.uint8)
    groundtruth2 = groundtruth2.astype(np.uint8)
    prediction1 = prediction1.astype(np.uint8)
    prediction2 = prediction2.astype(np.uint8)

    # Initialize HD95 and MSD values
    hd95_value1 = msd_value1 = hd95_value2 = msd_value2 = np.nan

    # Check if groundtruth1 and prediction1 contain any foreground (1s)
    if np.any(groundtruth1) and np.any(prediction1):
        hd95_value1 = hd95(prediction1, groundtruth1, voxelspacing=voxel_spacing)
        msd_value1 = assd(prediction1, groundtruth1, voxelspacing=voxel_spacing)
    else:
        print(f"Skipping HD95/MSD for label 1 (GTVp) due to empty groundtruth or prediction for this label.")

    # Check if groundtruth2 and prediction2 contain any foreground (1s)
    if np.any(groundtruth2) and np.any(prediction2):
        hd95_value2 = hd95(prediction2, groundtruth2, voxelspacing=voxel_spacing)
        msd_value2 = assd(prediction2, groundtruth2, voxelspacing=voxel_spacing)
    else:
        print(f"Skipping HD95/MSD for label 2 (GTVn) due to empty groundtruth or prediction for this label.")

    return hd95_value1, msd_value1, hd95_value2, msd_value2

# Compute aggregate DSC
def compute_agg_dice(intermediate_results):
    aggregate_results = {}
    
    # Extracting relevant metrics for aggregation
    TP1s = [v["TP1"] for v in intermediate_results]
    TP2s = [v["TP2"] for v in intermediate_results]
    vol_sum1s = [v["vol_sum1"] for v in intermediate_results]
    vol_sum2s = [v["vol_sum2"] for v in intermediate_results]
    
    DSC1s = [v["DSC1"] for v in intermediate_results]
    DSC2s = [v["DSC2"] for v in intermediate_results]
    
    HD95_1s = [v["HD95_1"] for v in intermediate_results if not np.isnan(v["HD95_1"])]
    HD95_2s = [v["HD95_2"] for v in intermediate_results if not np.isnan(v["HD95_2"])]
    
    MSD_1s = [v["MSD_1"] for v in intermediate_results if not np.isnan(v["MSD_1"])]
    MSD_2s = [v["MSD_2"] for v in intermediate_results if not np.isnan(v["MSD_2"])]

    # Aggregated DSC calculation for GTVp and GTVn
    DSCagg1 = 2 * np.sum(TP1s) / np.sum(vol_sum1s) if np.sum(vol_sum1s) > 0 else np.nan
    DSCagg2 = 2 * np.sum(TP2s) / np.sum(vol_sum2s) if np.sum(vol_sum2s) > 0 else np.nan
    
    # Conventional DSC aggregation
    aggregate_results['ConventionalDsc'] = {
        'GTVp': {
            'mean': np.nanmean(DSC1s),
            'median': np.nanmedian(DSC1s),
            'percentile_25': np.nanpercentile(DSC1s, 25),
            'percentile_75': np.nanpercentile(DSC1s, 75)
        },
        'GTVn': {
            'mean': np.nanmean(DSC2s),
            'median': np.nanmedian(DSC2s),
            'percentile_25': np.nanpercentile(DSC2s, 25),
            'percentile_75': np.nanpercentile(DSC2s, 75)
        }
    }

    # Aggregated DSC (DSCagg1 and DSCagg2) for GTVp and GTVn
    aggregate_results['AggregatedDsc'] = {
        'GTVp': {
            'mean': DSCagg1,
            'median': DSCagg1,  # Median is set to the same as the mean since it's a single value
            'percentile_25': DSCagg1,  # Percentiles don't apply to a single value
            'percentile_75': DSCagg1
        },
        'GTVn': {
            'mean': DSCagg2,
            'median': DSCagg2,
            'percentile_25': DSCagg2,
            'percentile_75': DSCagg2
        }
    }

    # HD95 aggregation with same structure
    aggregate_results['HD95'] = {
        'GTVp': {
            'mean': np.nanmean(HD95_1s),
            'median': np.nanmedian(HD95_1s),
            'percentile_25': np.nanpercentile(HD95_1s, 25),
            'percentile_75': np.nanpercentile(HD95_1s, 75)
        },
        'GTVn': {
            'mean': np.nanmean(HD95_2s),
            'median': np.nanmedian(HD95_2s),
            'percentile_25': np.nanpercentile(HD95_2s, 25),
            'percentile_75': np.nanpercentile(HD95_2s, 75)
        }
    }

    # MSD aggregation with same structure
    aggregate_results['MSD'] = {
        'GTVp': {
            'mean': np.nanmean(MSD_1s),
            'median': np.nanmedian(MSD_1s),
            'percentile_25': np.nanpercentile(MSD_1s, 25),
            'percentile_75': np.nanpercentile(MSD_1s, 75)
        },
        'GTVn': {
            'mean': np.nanmean(MSD_2s),
            'median': np.nanmedian(MSD_2s),
            'percentile_25': np.nanpercentile(MSD_2s, 25),
            'percentile_75': np.nanpercentile(MSD_2s, 75)
        }
    }

    return aggregate_results
# Compute individual metrics for each patient
def get_intermediate_metrics(patient_ID, groundtruth, prediction):
    overlap_measures = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures.SetNumberOfThreads(1)
    overlap_measures.Execute(groundtruth, prediction)

    DSC1 = overlap_measures.GetDiceCoefficient(1)
    DSC2 = overlap_measures.GetDiceCoefficient(2)

    vol_gt1, vol_gt2 = compute_volumes(groundtruth)
    vol_pred1, vol_pred2 = compute_volumes(prediction)

    vol_sum1 = vol_gt1 + vol_pred1
    vol_sum2 = vol_gt2 + vol_pred2
    TP1 = DSC1 * (vol_sum1) / 2
    TP2 = DSC2 * (vol_sum2) / 2

    # Get voxel spacing from the ground truth image
    voxel_spacing = groundtruth.GetSpacing()

    # Compute HD95 and MSD for both GTVp (label 1) and GTVn (label 2)
    hd95_value1, msd_value1, hd95_value2, msd_value2 = compute_surface_distances(groundtruth, prediction, voxel_spacing)

    # Return all metrics including the new distance metrics
    return {
        "PatientID": patient_ID,
        "TP1": TP1,
        "TP2": TP2,
        "vol_sum1": vol_sum1,
        "vol_sum2": vol_sum2,
        "DSC1": DSC1,
        "DSC2": DSC2,
        "vol_gt1": vol_gt1,
        "vol_gt2": vol_gt2,
        "HD95_1": hd95_value1,
        "HD95_2":hd95_value2,
        "MSD_1": msd_value1, 
        "MSD_2": msd_value2
    }

# Check the prediction for accuracy
def check_prediction(patient_ID, groundtruth, prediction):
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(sitk.sitkUInt8)
    caster.SetNumberOfThreads(1)
    groundtruth = caster.Execute(groundtruth)
    prediction = caster.Execute(prediction)

    stats = sitk.LabelStatisticsImageFilter()
    stats.Execute(prediction, prediction)
    labels = stats.GetLabels()
    if not all([l in [0, 1, 2] for l in labels]):
        raise RuntimeError(f"Patient {patient_ID}: Error. The labels are incorrect. They should be background: 0, GTVp: 1, GTVn: 2.")

    if not np.allclose(groundtruth.GetSpacing(), prediction.GetSpacing(), atol=0.000001):
        print(f"Patient {patient_ID}: Warning. Resolution mismatch between prediction and ground truth.")

    needs_resampling = (
        prediction.GetSize() != groundtruth.GetSize() or
        not np.allclose(prediction.GetDirection(), groundtruth.GetDirection(), atol=0.000001) or
        not np.allclose(prediction.GetOrigin(), groundtruth.GetOrigin(), atol=0.000001)
    )

    if needs_resampling:
        print(f"Patient {patient_ID}: Resampling prediction to match ground truth...")
        resample = sitk.ResampleImageFilter()
        resample.SetSize(groundtruth.GetSize())
        resample.SetOutputDirection(groundtruth.GetDirection())
        resample.SetOutputOrigin(groundtruth.GetOrigin())
        resample.SetOutputSpacing(groundtruth.GetSpacing())
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        prediction = resample.Execute(prediction)
    else:
        print(f'Patient {patient_ID}: No resampling needed.')

    return prediction

# Multithreaded evaluation function
def evaluate_patient(prediction_file, groundtruth_folder, chill):
    # Extract the filename (including patient ID) from the prediction file path
    patient_file = os.path.split(prediction_file)[-1]
    patient_ID = patient_file.split('_')[0]

    # Construct the ground truth file path by matching the filename
    gt_file = os.path.join(groundtruth_folder, patient_file)

    # If the corresponding ground truth file doesn't exist, handle based on chill flag
    if not os.path.exists(gt_file):
        if not chill:
            raise RuntimeError(f"Patient {patient_ID}: Ground truth file {gt_file} not found. Exiting due to mismatch.")
        else:
            print(f"Patient {patient_ID}: Ground truth file {gt_file} not found, skipping.")
            return None

    # Read the prediction and ground truth images
    prediction = sitk.ReadImage(prediction_file)
    groundtruth = sitk.ReadImage(gt_file)

    # Ensure prediction is properly resampled if needed
    prediction = check_prediction(patient_ID, groundtruth, prediction)

    return get_intermediate_metrics(patient_ID, groundtruth, prediction)

# Main function
def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation performance with DSC, HD95, MSD metrics.")
    parser.add_argument('groundtruth_folder', type=str, help='Path to the folder containing groundtruth masks')
    parser.add_argument('prediction_folder', type=str, help='Path to the folder containing prediction masks')
    parser.add_argument('--chill', action='store_true', default=False, help='Allow missing patients in prediction folder without raising error')

    args = parser.parse_args()
    print(os.listdir(args.prediction_folder))
    # Get prediction and ground truth files
    prediction_files = [os.path.join(args.prediction_folder, file) for file in os.listdir(args.prediction_folder) if "nii.gz" in file]

    intermediate_results = []

    # Multithreading to evaluate patients in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(evaluate_patient, f, args.groundtruth_folder, args.chill) for f in prediction_files]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:  # Only append non-None results
                intermediate_results.append(result)

    # Aggregate results
    aggregated_results = compute_agg_dice(intermediate_results)

    # Save individual patient results and aggregated metrics
    output_summary = {
        'patient_results': intermediate_results,
        'aggregate_results': aggregated_results
    }

    # Write results to summary.json in the prediction folder
    output_json_path = os.path.join(args.prediction_folder, 'summary_agg.json')
    with open(output_json_path, 'w') as json_file:
        json.dump(output_summary, json_file, indent=4)

    print(f"Results saved to {output_json_path}")

# Run the script
if __name__ == '__main__':
    main()
