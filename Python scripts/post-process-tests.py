import os
import nibabel as nib
import matplotlib.pyplot as plt
import torch
import SimpleITK as sitk
import json
import pandas as pd
import numpy as np
import pathlib as plb
import cc3d
import csv
import sys

### DEFINE FUNCTIONS ###

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

def compute_inf_results(inference_folder, inference_images, GT_labels_folder, GT_images, output_df):
    # compute challenge metrics on inference images and save results as csv 

    path_names = []
    dice = []
    fp= []
    fn = []
    for i in range(len(inference_images)):
        print(i)
        nii_gt_path, nii_pred_path = os.path.join(GT_labels_folder,inference_images[i]),os.path.join(inference_folder,inference_images[i])
        nii_gt_path = plb.Path(nii_gt_path)
        nii_pred_path = plb.Path(nii_pred_path)
        dice_sc, false_pos_vol, false_neg_vol = compute_metrics(nii_gt_path, nii_pred_path)

        csv_header = ['gt_name', 'dice_sc', 'false_pos_vol', 'false_neg_vol']
        path_names.append(nii_gt_path.name)
        dice.append(dice_sc)
        fp.append(false_pos_vol)
        fn.append(false_neg_vol)
    
    metrics_df = pd.DataFrame()
    metrics_df['image'],metrics_df['DICE'],metrics_df['challenge_FP'],metrics_df['challenge_FN'] = path_names,dice,fp,fn
    
    metrics_df.to_csv(output_df)

def process_array(arr,min_size):
    min_size = int(min_size)
    
    unique_elements, counts = np.unique(arr, return_counts=True)

    # Create a mapping to change values less than 125 to 0
    change_to_zero = unique_elements[counts < min_size]

    # Create a new array with the same values as the input array
    processed_arr = arr.copy()

    # Change values less than 125 to 0
    for val in change_to_zero:
        processed_arr[processed_arr == val] = 0
        
    processed_arr[processed_arr != 0] = 1

    return processed_arr

def post_process_nifti(nifti_pred_path,output_path,min_size):
    # get numpy array
    pred_array, voxel_vol = nii2numpy(nifti_pred_path)
    pred_conn_comp = con_comp(pred_array)
    
    # returns numpy array with removal of small connected components
    processed_arr = process_array(pred_conn_comp,min_size)
    
    # return and save nifti image
    nifti_img = nib.Nifti1Image(processed_arr, affine=np.eye(4)) 
    nib.save(nifti_img, output_path)

### DO POST-PROCESSING ###

i = int(sys.argv[1]) # depending on array job - should be 5, 10, 20, 40, or 80

processed_inference_folder='nnUNet_results/Dataset219_PETCT/nnUNetTrainer_1500epochs_NoMirroring__nnUNetPlans__3d_fullres/processedinferTs'+str(i)
os.mkdir(processed_inference_folder)


inference_folder = 'nnUNet_results/Dataset219_PETCT/nnUNetTrainer_1500epochs_NoMirroring__nnUNetPlans__3d_fullres/inferTs'
inference_images = [x for x in os.listdir(inference_folder) if '.nii.gz' in x]

GT_labels_folder = 'nnUnet_raw/Dataset220_PETCT/labelsTs'
GT_images = os.listdir(GT_labels_folder)

for image in inference_images:
    post_process_nifti(os.path.join(inference_folder,image),os.path.join(processed_inference_folder,image),i)

print('Post-processing done and images saved to correct folder')

### CALCULATE AND SAVE METRICS ###

path_names = []
dice = []
fp= []
fn = []
for i in range(len(inference_images)):
    print(i)
    nii_gt_path, nii_pred_path = os.path.join(GT_labels_folder,inference_images[i]),os.path.join(processed_inference_folder,inference_images[i])
    nii_gt_path = plb.Path(nii_gt_path)
    nii_pred_path = plb.Path(nii_pred_path)
    dice_sc, false_pos_vol, false_neg_vol = compute_metrics(nii_gt_path, nii_pred_path)

    csv_header = ['gt_name', 'dice_sc', 'false_pos_vol', 'false_neg_vol']
    path_names.append(nii_gt_path.name)
    dice.append(dice_sc)
    fp.append(false_pos_vol)
    fn.append(false_neg_vol)

print('Metrics calculated')

pp_metrics_df = pd.DataFrame()
pp_metrics_df['image'],pp_metrics_df['DICE'],pp_metrics_df['challenge_FP'],pp_metrics_df['challenge_FN'] = path_names,dice,fp,fn

pp_metrics_df.to_csv(str(i)+'_pp_metrics_df')

print('Metrics df saved')
