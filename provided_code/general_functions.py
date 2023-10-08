import random
import cv2
import torch
import os
import pathlib
import numpy as np
import pandas as pd


def load_file(file_name):   #  'D:\\Medical Physics\\xu\\code\\KBP-dose-prediction-main/data/train-pats/pt_127/Larynx.csv'
    """Load a file in one of the formats provided in the OpenKBP dataset
    :param file_name: the name of the file to be loaded
    :return: the file loaded
    """
    # Load the file as a csv
    loaded_file_df = pd.read_csv(file_name, index_col=0)  # 读取一个病人的一个csv文件
    # If the csv is voxel dimensions read it with numpy
    if 'voxel_dimensions.csv' in file_name:
        loaded_file = np.loadtxt(file_name)
    # Check if the data has any values
    elif loaded_file_df.isnull().values.any():    # df.isnull().any()会判断哪些列包含缺失值
        # Then the data is a vector, which we assume is for a mask of ones
        loaded_file = np.array(loaded_file_df.index).squeeze()
    else:
        # Then the data is a matrix of indices and data points
        loaded_file = {'indices': np.array(loaded_file_df.index).squeeze(),
                       'data': np.array(loaded_file_df['data']).squeeze()}

    return loaded_file


def get_paths(directory_path, ext=''):
    """Get the paths of every file with a specified extension in a directory
    :param directory_path: the path of the directory of interest
    :param ext: the extensions of the files of interest
    :return: the path of all files of interest
    """
    # if dir_name doesn't exist return an empty array
    if not os.path.isdir(directory_path):
        return []
    # Otherwise dir_name exists and function returns contents name(s)
    else:
        all_image_paths = []
        # If no extension given, then get all files
        if ext == '': 
            dir_list = os.listdir(directory_path)
            for iPath in dir_list: 
                if '.' != iPath[0]:  # Ignore hidden files 
                    all_image_paths.append('{}/{}'.format(directory_path, str(iPath)))
        else:
            # Get list of paths for files with the extension ext
            data_root = pathlib.Path(directory_path) 
            for iPath in data_root.glob('*.{}'.format(ext)): 
                all_image_paths.append(str(iPath).replace("\\", "/")) 

    return all_image_paths


def get_paths_from_sub_directories(main_directory_path, sub_dir_list, ext=''):
    """Compiles a list of all paths within each sub directory listed in sub_dir_list that follows the main_dir_path
    :param main_directory_path: the path for the main directory of interest
    :param sub_dir_list: the name(s) of the directory of interest that are in the main_directory
    :param ext: the extension of the files of interest (in the usb directories)
    :return:
    """
    # Initialize list of paths
    path_list = []
    # Iterate through the sub directory names and build up the path list
    for sub_dir in sub_dir_list:
        paths_to_add = get_paths('{}/{}'.format(main_directory_path, sub_dir), ext=ext)
        path_list.extend(paths_to_add)

    return path_list


def sparse_vector_function(x, indices=None):   # (128,128,6,128)
    """Convert a tensor into a dictionary of the non zero values and their corresponding indices
    :param x: the tensor or, if indices is not None, the values that belong at each index
    :param indices: the raveled indices of the tensor
    :return:  sparse vector in the form of a dictionary
    """
    if indices is None:
        y = {'data': x[x > 0], 'indices': np.nonzero(x.flatten())[-1]}
        # print(y)
    else:
        y = {'data': x[x > 0], 'indices': indices[x > 0]}
    return y


def make_directory_and_return_path(dir_path):
    """Makes a directory only if it does not already exist
    :param dir_path: the path of the directory to be made
    :return: returns the directory path
    """
    os.makedirs(dir_path, exist_ok=True)

    return dir_path



# Random flip
def random_flip_3d(list_images, list_axis=(0, 1, 2), p=0.5):
    if random.random() <= p:
        if 0 in list_axis:
            if random.random() <= 0.5:
                for image_i in range(len(list_images)):
                    list_images[image_i] = list_images[image_i][:, ::-1, :, :]
        if 1 in list_axis:
            if random.random() <= 0.5:
                for image_i in range(len(list_images)):
                    list_images[image_i] = list_images[image_i][:, :, ::-1, :]
        if 2 in list_axis:
            if random.random() <= 0.5:
                for image_i in range(len(list_images)):
                    list_images[image_i] = list_images[image_i][:, :, :, ::-1]

    return list_images


# Random rotation using OpenCV
def random_rotate_around_z_axis(list_images,
                                list_angles,
                                list_interp,
                                list_boder_value,
                                p=0.5):
    if random.random() <= p:
        # Randomly pick an angle list_angles
        _angle = random.sample(list_angles, 1)[0]
        # Do not use random scaling, set scale factor to 1
        _scale = 1.

        for image_i in range(len(list_images)):
            for chan_i in range(list_images[image_i].shape[0]):
                for slice_i in range(list_images[image_i].shape[1]):
                    rows, cols = list_images[image_i][chan_i, slice_i, :, :].shape
                    M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), _angle, scale=_scale)
                    list_images[image_i][chan_i, slice_i, :, :] = \
                        cv2.warpAffine(list_images[image_i][chan_i, slice_i, :, :],
                                       M,
                                       (cols, rows),
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=list_boder_value[image_i],
                                       flags=list_interp[image_i])
    return list_images


# Random translation
def random_translate(list_images, roi_mask, p, max_shift, list_pad_value):
    if random.random() <= p:
        exist_mask = np.where(roi_mask > 0)
        ori_z, ori_h, ori_w = list_images[0].shape[1:]

        bz = min(max_shift - 1, np.min(exist_mask[0]))
        ez = max(ori_z - 1 - max_shift, np.max(exist_mask[0]))
        bh = min(max_shift - 1, np.min(exist_mask[1]))
        eh = max(ori_h - 1 - max_shift, np.max(exist_mask[1]))
        bw = min(max_shift - 1, np.min(exist_mask[2]))
        ew = max(ori_w - 1 - max_shift, np.max(exist_mask[2]))

        for image_i in range(len(list_images)):
            list_images[image_i] = list_images[image_i][:, bz:ez + 1, bh:eh + 1, bw:ew + 1]

        # Pad to original size
        list_images = random_pad_to_size_3d(list_images,
                                            target_size=[ori_z, ori_h, ori_w],
                                            list_pad_value=list_pad_value)
    return list_images


# To tensor, images should be C*Z*H*W
def to_tensor(list_images):
    for image_i in range(len(list_images)):
        list_images[image_i] = torch.from_numpy(list_images[image_i].copy()).float()
    return list_images


# Pad
def random_pad_to_size_3d(list_images, target_size, list_pad_value):
    _, ori_z, ori_h, ori_w = list_images[0].shape[:]
    new_z, new_h, new_w = target_size[:]

    pad_z = new_z - ori_z
    pad_h = new_h - ori_h
    pad_w = new_w - ori_w

    pad_z_1 = random.randint(0, pad_z)
    pad_h_1 = random.randint(0, pad_h)
    pad_w_1 = random.randint(0, pad_w)

    pad_z_2 = pad_z - pad_z_1
    pad_h_2 = pad_h - pad_h_1
    pad_w_2 = pad_w - pad_w_1

    output = []
    for image_i in range(len(list_images)):
        _image = list_images[image_i]
        output.append(np.pad(_image,
                             ((0, 0), (pad_z_1, pad_z_2), (pad_h_1, pad_h_2), (pad_w_1, pad_w_2)),
                             mode='constant',
                             constant_values=list_pad_value[image_i])
                      )
    return output
