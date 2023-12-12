import pandas
import os
import nibabel as nib
import matplotlib.pyplot as plt
import torch
import SimpleITK as sitk
import json
import pandas as pd
import numpy as np
import numpy as np
import pathlib as plb
import cc3d
import csv
import sys

#path_to_results219="nnUNet_results/Dataset219_PETCT/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessed/summary.json"
#path_to_results219="nnUNet_results/Dataset219_PETCT/nnUNetTrainer_probabilisticOversampling_033__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2/postprocessed/summary.json"
#path_to_results219="nnUNet_results/Dataset219_PETCT/nnUNetTrainer_1500epochs_1e2lr__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessed/summary.json"
path_to_results219="nnUNet_results/Dataset219_PETCT/nnUNetTrainer__nnUNetPlans__3d_fullres_maxnum_512_patch_192/crossval_results_folds_0_1_2/postprocessed/summary.json"

def make_results_df(path_to_summary_json): 
    with open(path_to_summary_json, "r") as read_file:
            dict_results = json.load(read_file)

    metrics_per_case = dict_results['metric_per_case']
    results = []
    for case in metrics_per_case:
        metrics = case['metrics']['1']
        row = {
            'DICE': metrics['Dice'],
            'FN': metrics['FN'],
            'FP': metrics['FP'],
            'IoU': metrics['IoU'],
            'TN': metrics['TN'],
            'TP': metrics['TP'],
            'n_pred': metrics['n_pred'],
            'n_ref': metrics['n_ref'],
            'prediction_file': case['prediction_file'],
            'reference_file': case['reference_file']
            }
        results.append(row)

    df = pd.DataFrame(results)
    
    return df

def nii2numpy(nii_path):
    # input: path of NIfTI segmentation file, output: corresponding numpy array and voxel_vol in ml
    mask_nii = nib.load(str(nii_path))
    mask = mask_nii.get_fdata()
    pixdim = mask_nii.header['pixdim']   
    voxel_vol = pixdim[1]*pixdim[2]*pixdim[3]/1000
    return mask, voxel_vol


def con_comp(seg_array):
    # input: a binary segmentation array output: an array with seperated (indexed) connected components of the segmentation array
    connectivity = 18
    conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
    return conn_comp


def false_pos_pix(gt_array,pred_array):
    # compute number of voxels of false positive connected components in prediction mask
    pred_conn_comp = con_comp(pred_array)
    
    false_pos = 0
    for idx in range(1,pred_conn_comp.max()+1):
        comp_mask = np.isin(pred_conn_comp, idx)
        if (comp_mask*gt_array).sum() == 0:
            false_pos = false_pos+comp_mask.sum()
    return false_pos



def false_neg_pix(gt_array,pred_array):
    # compute number of voxels of false negative connected components (of the ground truth mask) in the prediction mask
    gt_conn_comp = con_comp(gt_array)
    
    false_neg = 0
    for idx in range(1,gt_conn_comp.max()+1):
        comp_mask = np.isin(gt_conn_comp, idx)
        if (comp_mask*pred_array).sum() == 0:
            false_neg = false_neg+comp_mask.sum()
            
    return false_neg


def dice_score(mask1,mask2):
    # compute foreground Dice coefficient
    overlap = (mask1*mask2).sum()
    sum = mask1.sum()+mask2.sum()
    dice_score = 2*overlap/sum
    return dice_score



def compute_metrics(nii_gt_path, nii_pred_path):
    # main function
    gt_array, voxel_vol = nii2numpy(nii_gt_path)
    pred_array, voxel_vol = nii2numpy(nii_pred_path)

    false_neg_vol = false_neg_pix(gt_array, pred_array) *voxel_vol
    false_pos_vol = false_pos_pix(gt_array, pred_array) *voxel_vol
    dice_sc = dice_score(gt_array,pred_array)

    return dice_sc, false_pos_vol, false_neg_vol

df = make_results_df(path_to_results219)

df['challenge_DICE'], df['challenge_FP'],df['challenge_FN'] = zip(*df.apply(lambda row: compute_metrics(row['reference_file'], row['prediction_file']), axis=1))

df.to_csv('d-219-CV-all-results.csv')
