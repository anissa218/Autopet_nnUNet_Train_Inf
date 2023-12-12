import numpy as np
import nibabel as nib
import pathlib as plb
import cc3d
import csv
import sys
import os
import pandas as pd

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

def find_num_conn_comps(path):
    array, voxel_vol = nii2numpy(path)
    conn_comp = con_comp(array)
    num_comps = np.max(conn_comp)
    
    return num_comps

CV_images_folder = 'nnUNet_results/Dataset219_PETCT/nnUNetTrainer_1500epochs_1e2lr__nnUNetPlans__3d_fullres_maxnum_512_patch_192/crossval_results_folds_0_1_2_3_4'
CV_images = [x for x in os.listdir(CV_images_folder) if '.nii.gz' in x]
processed_inference_folder='nnUNet_results/Dataset219_PETCT/nnUNetTrainer_1500epochs_1e2lr__nnUNetPlans__3d_fullres_maxnum_512_patch_192/processedCV'

GT_labels_folder = 'nnUnet_raw/Dataset219_PETCT/labelsTr'

path_names = []
dice = []
fp= []
fn = []
for i in range(len(CV_images)):
    print(i)
    nii_gt_path, nii_pred_path = os.path.join(GT_labels_folder,CV_images[i]),os.path.join(processed_inference_folder,CV_images[i])
    nii_gt_path = plb.Path(nii_gt_path)
    nii_pred_path = plb.Path(nii_pred_path)
    dice_sc, false_pos_vol, false_neg_vol = compute_metrics(nii_gt_path, nii_pred_path)

    csv_header = ['gt_name', 'dice_sc', 'false_pos_vol', 'false_neg_vol']
    path_names.append(nii_gt_path.name)
    dice.append(dice_sc)
    fp.append(false_pos_vol)
    fn.append(false_neg_vol)

c_pp_metrics_df = pd.DataFrame()
c_pp_metrics_df['image'],c_pp_metrics_df['DICE'],c_pp_metrics_df['challenge_FP'],c_pp_metrics_df['challenge_FN'] = path_names,dice,fp,fn

c_pp_metrics_df.to_csv('c-219-pp-CV-results.csv')