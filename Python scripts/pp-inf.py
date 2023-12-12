import numpy as np
import nibabel as nib
import pathlib as plb
import cc3d
import csv
import sys
import os
import matplotlib.pyplot as plt
import torch
import SimpleITK as sitk
import json
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

def compute_inf_results(infernece_folder, inference_images, GT_labels_folder, GT_images, output_df):
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

def load_process_inf_metrics_from_df(input_df):
    f_metrics_df = input_df
    f_metrics_df['subject_id'] = f_metrics_df['image'].apply(lambda x: x.split('_')[0]+"_"+x.split('_')[1])
    f_metrics_df['subject_date'] = f_metrics_df['image'].apply(lambda x: x.split('_')[2].split(".")[0])
    
    #f_metrics_df = f_metrics_df.drop(columns=['Unnamed: 0'])
    
    # Merge the two DataFrames based on 'subject_id' and 'subject_date'
    merged_df = f_metrics_df.merge(clinical_data_df, left_on=['subject_id', 'subject_date'], right_on=['Subject ID', 'Study Date'], how='left')

    # Drop the duplicate columns 'Subject ID' and 'Study Date' (optional, if you want to keep only one set of them)
    merged_df.drop(columns=['Subject ID', 'Study Date'], inplace=True)

    # Rename the 'Diagnosis' column to 'diagnosis' (optional, if you want to match the column name)
    merged_df.rename(columns={'Diagnosis': 'diagnosis'}, inplace=True)
    
    return merged_df

# COMPUTE INF RESULTS (BEFORE PP)
inference_folder = 'nnUNet_results/Dataset219_PETCT/nnUNetTrainer_1500epochs__nnUNetPlans__3d_fullres/inferTs'

GT_labels_folder = 'nnUnet_raw/Dataset220_PETCT/labelsTs'

GT_images = os.listdir(GT_labels_folder)
inference_images = [x for x in os.listdir(inference_folder) if '.nii.gz' in x]
compute_inf_results(inference_folder,inference_images,GT_labels_folder,GT_images,'h_219_inf_results.csv')
print('calculated and saved inf results')

# DO PP ON INF AND CALCULATE RESULTS

# dict_of_paths = {'a':['a_219_inf_results.csv','nnUNetTrainer_1500epochs_NoMirroring__nnUNetPlans__3d_fullres']}
# dict_of_paths['default']=['default_219_inf_results.csv','nnUNetTrainer__nnUNetPlans__3d_fullres']
# dict_of_paths['f'] = ['f_219_inf_results.csv','nnUNetTrainer_probabilisticOversampling_033__nnUNetPlans__3d_fullres']
# dict_of_paths['e'] = ['e_219_inf_results.csv','nnUNetTrainer_1500epochs_1e2lr__nnUNetPlans__3d_fullres']
# dict_of_paths['d'] = ['d_219_inf_results.csv','nnUNetTrainer__nnUNetPlans__3d_fullres_maxnum_512_patch_192']
# dict_of_paths['c'] = ['c_219_inf_results.csv','nnUNetTrainer_1500epochs_1e2lr__nnUNetPlans__3d_fullres_maxnum_512_patch_192']
dict_of_paths = {'h':['h_219_inf_results.csv','nnUNetTrainer_1500epochs__nnUNetPlans__3d_fullres']}

from datetime import datetime

clinical_data_df = pd.read_csv('Clinical Metadata FDG PET_CT Lesions (1).csv')
#clinical_data_df['Study Date'] = clinical_data_df['Study Date'].apply(lambda x: x.replace("/", ""))

# add zeros to date so it is compatible w/ other data
clinical_data_df['Study Date'] = clinical_data_df['Study Date'].apply(lambda x: (datetime.strptime(x, "%m/%d/%Y")).strftime("%m%d%Y"))

GT_labels_folder = 'nnUnet_raw/Dataset220_PETCT/labelsTs'
base_path = 'nnUNet_results/Dataset219_PETCT/'

# iterate through each model
for key in dict_of_paths.keys():
    print(key)
    relative_path = dict_of_paths[key][1]

    # create folder for pp images
    processed_inference_folder = os.path.join(base_path,relative_path,'pp10_inferTs')
    if not os.path.exists(processed_inference_folder):
        os.mkdir(processed_inference_folder)

    # get non-processed inference images
    inference_folder = os.path.join(base_path,relative_path,'inferTs')
    inference_images = [x for x in os.listdir(inference_folder) if '.nii.gz' in x]

    # apply post processing and save images to folder
    for image in inference_images:
        post_process_nifti(os.path.join(inference_folder,image),os.path.join(processed_inference_folder,image),10)

    print('post-processing finished')
    
    # calculate new metrics and save results
    path_names = []
    dice = []
    fp= []
    fn = []
    for i in range(len(inference_images)):
        nii_gt_path, nii_pred_path = os.path.join(GT_labels_folder,inference_images[i]),os.path.join(processed_inference_folder,inference_images[i])
        nii_gt_path = plb.Path(nii_gt_path)
        nii_pred_path = plb.Path(nii_pred_path)
        dice_sc, false_pos_vol, false_neg_vol = compute_metrics(nii_gt_path, nii_pred_path)
    
        csv_header = ['gt_name', 'dice_sc', 'false_pos_vol', 'false_neg_vol']
        path_names.append(nii_gt_path.name)
        dice.append(dice_sc)
        fp.append(false_pos_vol)
        fn.append(false_neg_vol)

    print('metrics calculated')
    
    pp_metrics_df = pd.DataFrame()
    pp_metrics_df['image'],pp_metrics_df['DICE'],pp_metrics_df['challenge_FP'],pp_metrics_df['challenge_FN'] = path_names,dice,fp,fn

    # add column in df for diagnosis
    pp_metrics_df = load_process_inf_metrics_from_df(pp_metrics_df)

    print('made df and merged with clinical data df')
    
    df_name = str(key) + '-219-10pp-CV-results.csv'
    pp_metrics_df.to_csv(df_name)

    print('metrics saved as df')
