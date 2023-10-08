from SimpleITK.SimpleITK import Gradient
from SimpleITK.extra import ReadImage
import torch
from torch.utils import data
import numpy as np
from torchio.transforms import transform
from provided_code.general_functions import get_paths, load_file
import itk
import random
import os
import time
from skimage import measure, feature
import trimesh
# import cv2
from PIL import Image
import SimpleITK as sitk
from sklearn import neighbors
from matplotlib import pyplot as plt
import random
import pdb
import math
import shutil
from pathlib import Path
import cv2
import torchio
import SimpleITK as sitk

dist_max_ct = 51
dist_max_oar = 49

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class KBPDataset(data.Dataset):
    """Generates data for tensorflow"""

    def __init__(self, file_paths_list, flipped=False, rotate=False, noise=False, deformation=False, patient_shape=(128, 128, 128), ct_range=[-400, 2400],
                 shuffle=True, mode_name='training_model'):
        """Initialize the DataLoader class, which loads the data for OpenKBP
        :param file_paths_list: list of the directories or single files where data for each patient is stored
        :param batch_size: the number of data points to load in a single batch
        :param patient_shape: the shape of the patient data
        :param shuffle: whether or not order should be randomized
        """
        # Set file_loader specific attributes
        self.rois = dict(oars=['Brainstem', 'SpinalCord', 'RightParotid', 'LeftParotid',
                               'Esophagus', 'Larynx', 'Mandible'], targets=['PTV56', 'PTV63', 'PTV70'])

        self.patient_shape = patient_shape  # Shape of the patient
        self.min_ct = ct_range[0]
        self.max_ct = ct_range[1]
        self.rotate_angle_list = [i for i in range(-2,-20,-2)] + [i for i in range(0,20,2)]
        self.file_paths_list = []
        self.patient_id_list = []
        self.if_flipped = flipped
        self.if_rotate = rotate
        self.if_noise = noise
        self.if_deformation = deformation

        # original patient
        for j in range(len(file_paths_list)):
            if 'pt_38' in file_paths_list[j]:
                continue
            self.file_paths_list.append([file_paths_list[j]]) 
            self.patient_id_list.append(
                'pt_{}'.format(file_paths_list[j].split('/pt_')[1].split('/')[0].split('.csv')[0]))

        # Indicator as to whether or not data is shuffled
        self.shuffle = shuffle  

        # make a list of all rois:['Brainstem', 'SpinalCord', 'RightParotid', 'LeftParotid','Esophagus', 'Larynx', 'Mandible','PTV56', 'PTV63', 'PTV70']
        self.full_roi_list = sum(map(list, self.rois.values()), [])  
        self.num_rois = len(self.full_roi_list)  # 10

        # Set files to be loaded
        self.required_files = None
        self.mode_name = mode_name  # Defines the mode for which data must be loaded for
        self.set_mode(self.mode_name)  # Set load mode to prediction by default


    def set_mode(self, mode_name, single_file_name=None):
        """Selects the type of data that is loaded
        :param mode_name: the name of the mode that the data loader is switching to
        :param single_file_name: the name of the file that should be loaded (only used if the mode_name is 'single_file')
        """
        self.mode_name = mode_name
        if mode_name == 'training_model':
            # The mode that should be used when training or validing a model
            self.required_files = {'dose': (self.patient_shape + (1,)),  # The shape of dose tensor
                                   'ct': (self.patient_shape + (1,)),  # The shape of ct tensor
                                   'structure_masks': (self.patient_shape + (self.num_rois,)),
                                   # The shape of the structure mask tensor
                                   'possible_dose_mask': (self.patient_shape + (1,)),
                                   # Mask of where dose can be deposited
                                   'voxel_dimensions': (3,)
                                   # Physical dimensions (in mm) of voxels
                                   }
        elif mode_name == 'dose_prediction':
            # The mode that should be used when training or validing a model
            self.required_files = {'ct': (self.patient_shape + (1,)),  # The shape of ct tensor
                                   'structure_masks': (self.patient_shape + (self.num_rois,)),
                                   # The shape of the structure mask tensor
                                   'possible_dose_mask': (self.patient_shape + (1,)),
                                   # Mask of where dose can be deposited
                                   'voxel_dimensions': (3,)  # Physical dimensions (in mm) of voxels
                                   }
            print('Warning: Batch size has been changed to 1 for dose prediction mode')
        elif mode_name == 'predicted_dose':
            # This mode loads a single feature (e.g., dose, masks for all structures)
            self.required_files = {mode_name: (self.patient_shape + (1,))}  # The shape of a dose tensor

        elif mode_name == 'evaluation':
            # The mode that should be used evaluate the quality of predictions
            self.required_files = {'dose': (self.patient_shape + (1,)),  # The shape of dose tensor
                                   'structure_masks': (self.patient_shape + (self.num_rois,)),
                                   'voxel_dimensions': (3,),  # Physical dimensions (in mm) of voxels
                                   'possible_dose_mask': (self.patient_shape + (1,)),
                                   }
            print('Warning: Batch size has been changed to 1 for evaluation mode')

        else:
            print('Mode does not exist. Please re-run with either \'training_model\', \'prediction\', '
                  '\'predicted_dose\', or \'evaluation\'')


    def __getitem__(self, index):
        """Generates a data sample containing batch_size samples X : (n_samples, *dim, n_channels)
        :param file_paths_to_load: the paths of the files to be loaded
        :return: a dictionary of all the loaded files
        """

        # Initialize dictionary for loaded data and lists to track patient path and ids 初始化
        data_sample = {}.fromkeys(self.required_files)  

        # Loop through each key in tf data to initialize the tensor with zeros
        for key in data_sample:
            # Make dictionary with appropriate data sizes for bath learning
            data_sample[key] = np.zeros(self.required_files[key], dtype=np.float32)  # dose:(128,128,128,1),'ct':(128,128,128,1),'structure':(128,128,128,10),'dose mask':(128,128,128,1),voxel dimension(3,)

        # Generate data
        pat_path = self.file_paths_list[index][0] 
        pat_id = pat_path.split('/')[-1].split('.')[0] 
        loaded_data_dict, ptv, oar, coarse_dose, directionmap = self.load_and_shape_data(pat_path)

        for key in data_sample:
            data_sample[key] = loaded_data_dict[key]

        data_sample["direction"] = directionmap * data_sample["possible_dose_mask"]
        data_sample["coarse_dose"] = coarse_dose * data_sample["possible_dose_mask"]
        data_sample["ct"] = data_sample["ct"] * data_sample["possible_dose_mask"]

        if 'dose' in data_sample:
            data_sample['dose'] = data_sample['dose'] * data_sample['possible_dose_mask']

        data_sample['patient_id'] = pat_id
        # store ptv oar numpy
        data_sample['ptv'] = ptv
        data_sample['oar'] = oar


        # flip
        if self.if_flipped:
            if random.randint(0,1) == 1:
                data_sample['patient_id'] = data_sample['patient_id'] + '_flipped'
                for key in data_sample:
                    if key != 'voxel_dimensions' and key != 'patient_id':
                        data_sample[key] = np.ascontiguousarray(np.flip(data_sample[key], axis=3))
        
        # rotate
        angle = self.rotate_angle_list[random.randint(0, len(self.rotate_angle_list) - 1)]
        if self.if_rotate:
            data_sample['patient_id'] = data_sample['patient_id'] + '_' + str(angle)
            for key in data_sample:
                if key != 'voxel_dimensions' and key != 'structure_masks' and key != 'patient_id' and key != "direction":
                    data_sample[key] = np.ascontiguousarray(self.rotate(data_sample[key], angle))
                    if len(data_sample[key].shape) == 3:
                        data_sample[key] = np.expand_dims(data_sample[key], axis=0)
                    
                elif key != 'voxel_dimensions' and key != 'patient_id' :
                    data_sample[key] = np.ascontiguousarray(self.structure_rotate(data_sample[key], angle))

        # noise
        if self.if_noise:
            data_sample['patient_id'] = data_sample['patient_id'] + '_noised'
            for key in data_sample:
                if key == 'ct':
                    data_sample[key] = np.ascontiguousarray(self.addnoise(data_sample[key]))
                    if len(data_sample[key].shape) == 3:
                        data_sample[key] = np.expand_dims(data_sample[key], axis=0)

        if self.if_deformation:
            data_sample['patient_id'] = data_sample['patient_id'] + '_deformed'
            transform = torchio.RandomElasticDeformation(num_control_points=(7, 7, 7), locked_borders=2)
            for key in data_sample:
                if key != 'voxel_dimensions' and key != 'patient_id':
                    data_sample[key] = transform(data_sample[key])
        
        # get patient path
        data_sample['patient_path'] = pat_path
        # print('check:',)

        # self.save(data_sample)

        return data_sample


    def __len__(self):
        return len(self.file_paths_list)

    def save(self, data_sample): 
        out = Path('/public/bme/home/v-tenglin/BDCLDosePrediction/Data/save/')
        # if os.path.exists(out):
        #     shutil.rmtree(out)  # delete output folder
        # os.makedirs(out)  # make new output folder

        spacing = tuple(data_sample['voxel_dimensions'][...,[2,1,0]])
        self.savetome(spacing, data_sample["direction"][0,...], "d1", data_sample['patient_id'])
        self.savetome(spacing, data_sample["direction"][1,...], "d2", data_sample['patient_id'])
        self.savetome(spacing, data_sample["direction"][2,...], "d3", data_sample['patient_id'])
        self.savetome(spacing, data_sample["direction"][3,...], "d4", data_sample['patient_id'])
        self.savetome(spacing, data_sample["direction"][4,...], "d5", data_sample['patient_id'])
        self.savetome(spacing, data_sample["direction"][5,...], "d6", data_sample['patient_id'])
        self.savetome(spacing, data_sample["direction"][6,...], "d7", data_sample['patient_id'])
        self.savetome(spacing, data_sample["direction"][7,...], "d8", data_sample['patient_id'])
        self.savetome(spacing, data_sample["direction"][8,...], "d9", data_sample['patient_id'])

        self.savetome(spacing,data_sample["ct"],'ct',data_sample['patient_id'])
        self.savetome(spacing,data_sample["coarse_dose"],'coarse_dose',data_sample['patient_id'])
        self.savetome(spacing,data_sample["possible_dose_mask"],'possible_dose_mask',data_sample['patient_id'])
        self.savetome(spacing,data_sample["structure_masks"][7,...],'PTV56',data_sample['patient_id'])
        self.savetome(spacing,data_sample["dose"],'dose',data_sample['patient_id'])

    def savetome(self, spacing, source, obj, id):
        # ctt = sitk.GetImageFromArray(np.flip(np.flip(np.squeeze(source.astype(float))),1))
        ctt = sitk.GetImageFromArray(np.squeeze(source.astype(float)))
        ctt.SetSpacing(spacing)
        sitk.WriteImage(ctt,
        '/public/bme/home/v-tenglin/BDCLDosePrediction/Data/save/{}_{}.nii.gz'.format(str(id), str(obj)))                                                                                


    def addnoise(self, img):
        img = sitk.GetImageFromArray(img)
        mean = np.random.uniform(0, 0.1)
        std = np.random.uniform(0, 0.1)
        noiseFilter = sitk.AdditiveGaussianNoiseImageFilter()
        noiseFilter.SetMean(mean)
        noiseFilter.SetStandardDeviation(std)
        return sitk.GetArrayFromImage(noiseFilter.Execute(img))


    def structure_rotate(self, img, angle):
        img = np.squeeze(img)
        idx = img.shape[0]
        leng = img.shape[1]
        struc_imgs = []
        for k in range(idx):
            list_img = []
            for i in range(leng):
                tem = img[k, i, :, :]
                img_t = Image.fromarray(tem)
                img_t = img_t.rotate(angle)
                img_arr = np.asarray(img_t)
                list_img.append(img_arr)
            struc_imgs.append(list_img)
        img_rota = np.array(struc_imgs)
        return img_rota

    def rotate(self, img, angle):
        img = np.squeeze(img)
        c = img.shape[0]
        list_img = []
        for i in range(c):
            tem = img[i, :, :]
            img_t = Image.fromarray(tem)
            img_t = img_t.rotate(angle)
            img_arr = np.asarray(img_t)
            list_img.append(img_arr)
        img_rota = np.array(list_img)
        return img_rota

    def translate(self, img, dosemask, max_shift, list_pad_value):
        exist_mask = np.where(dosemask > 0)
        ori_z, ori_h, ori_w = img.shape[1:]

        bz = min(max_shift - 1, np.min(exist_mask[0]))
        ez = max(ori_z - 1 - max_shift, np.max(exist_mask[0]))
        bh = min(max_shift - 1, np.min(exist_mask[1]))
        eh = max(ori_h - 1 - max_shift, np.max(exist_mask[1]))
        bw = min(max_shift - 1, np.min(exist_mask[2]))
        ew = max(ori_w - 1 - max_shift, np.max(exist_mask[2]))

        img = img[:, bz:ez + 1, bh:eh + 1, bw:ew + 1]

        # Pad to original size
        img = self.random_pad_to_size_3d(   img,
                                            target_size=[ori_z, ori_h, ori_w])
        return img

    def random_pad_to_size_3d(self, img, target_size):
        ori_z, ori_h, ori_w = img.shape[1:]
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

        output = np.pad(    img,
                            ((0, 0), (pad_z_1, pad_z_2), (pad_h_1, pad_h_2), (pad_w_1, pad_w_2)),
                            mode='constant',
                            constant_values=0)
        return output

    def load_and_shape_data(self, path_to_load):
        """ Reshapes data that is stored as vectors into matrices
        :param path_to_load: the path of the data that needs to be loaded. If the path is a directory, all data in the
         directory will be loaded. If path is a file then only that file will be loaded.
        :return: Loaded data with the appropriate shape
        """

        # Initialize the dictionary for the loaded files
        loaded_file = {}
        if '.csv' in path_to_load:
            loaded_file[self.mode_name] = load_file(path_to_load)
        else:
            files_to_load = get_paths(path_to_load, ext='')  # path_to_load是一个病人文件
            for f in files_to_load:
                f_name = f.split('/')[-1].split('.')[0]  # Brainstem
                if f_name in self.required_files or f_name in self.full_roi_list:
                    loaded_file[f_name] = load_file(f) 

        # Initialize matrices for features
        shaped_data = {}.fromkeys(self.required_files)
        for key in shaped_data:
            shaped_data[key] = np.zeros(self.required_files[key], dtype=np.float32)  # 4维

        # Populate matrices that were no initialized as []
        for key in shaped_data: 
            if key == 'structure_masks':  
                # Convert dictionary of masks into a tensor (necessary for tensorflow)
                for roi_idx, roi in enumerate(self.full_roi_list): 
                    if roi in loaded_file.keys(): 
                        np.put(shaped_data[key], self.num_rois * loaded_file[roi] + roi_idx, int(1))
                shaped_data[key] = np.ascontiguousarray(shaped_data[key].transpose((3, 2, 0, 1)))
            elif key == 'possible_dose_mask': 
                np.put(shaped_data[key], loaded_file[key], int(1))
                shaped_data[key] = np.ascontiguousarray(shaped_data[key].transpose((3, 2, 0, 1)))
            elif key == 'voxel_dimensions':  
                shaped_data[key] = loaded_file[key].copy() 
                shaped_data[key][0] = loaded_file[key][2]  # spacing z
                shaped_data[key][1] = loaded_file[key][0]  # spacing y
                shaped_data[key][2] = loaded_file[key][1]  # spacing x
            else:  # Files with shape
                np.put(shaped_data[key], loaded_file[key]['indices'], loaded_file[key]['data'])
                shaped_data[key] = np.ascontiguousarray(shaped_data[key].transpose((3, 2, 0, 1)))

                if key == 'ct':
                    img = shaped_data[key].copy()
                    img[img < self.min_ct] = self.min_ct  # (1,128,128,128)
                    img[img > self.max_ct] = self.max_ct
                    img = (img - self.min_ct) / (self.max_ct - self.min_ct)
                    shaped_data[key] = img

        # genearate the ptv image including ptv53, 63, 70
        ptv = shaped_data["structure_masks"][7,...]+shaped_data["structure_masks"][8,...]+shaped_data["structure_masks"][9,...]
        ptv = np.expand_dims(np.clip(ptv,0,1),0)

        oar = np.zeros_like(ptv)
        for i in range(7):
            oar += shaped_data["structure_masks"][i,...]
        oar = np.expand_dims(np.clip(oar,0,1),0)

        coarse_dose_root = "/public/bme/home/v-tenglin/BDCLDosePrediction/Data/coarse_data/"
        # coarse_dose = np.fliplr(np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(coarse_dose_root)),0))
        coarse_dose = np.fliplr(np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(coarse_dose_root + "pred_" + path_to_load.split("/")[-1] + ".nii.gz")),0))
        
        # coarse_dose = (coarse_dose - coarse_dose.min()) / (coarse_dose.max() - coarse_dose.min())

        directionmap = np.zeros((9,128,128,128))
        directions_root = "/public/bme/home/v-tenglin/BDCLDosePrediction/Data/direction/"
        for i in range(directionmap.shape[0]):
            directionmap[i,...] = sitk.GetArrayFromImage(sitk.ReadImage(directions_root + path_to_load.split("/")[-1] + "_d1"+"_{}".format(i+1) + ".nii.gz" ))

        '''
        directionmap = np.zeros((9,128,128,128))

        zmin = np.where(ptv[0,...] != 0)[0].min(); zmax = np.where(ptv[0,...] != 0)[0].max()
        for i in range(directionmap.shape[1]):
            if i <= zmin:
                ptvm = ptv[0,zmin,...].copy()
            elif i >= zmax:
                ptvm = ptv[0,zmax,...].copy()
            else:        
                ptvm = ptv[0,i,...].copy() 
            edges = feature.canny(ptvm, low_threshold=1, high_threshold=1)
            if edges.max() == False:
                print('max:',path_to_load.split("/")[-1])
            # if edges.min() == False:
            #     print('min:',path_to_load.split("/")[-1])
            directionmap[:,i,:,:] = self.drawline(directionmap[:,i,:,:], [0,40,80,120,160,200,240,280,320], edges, path_to_load.split("/")[-1])
        '''

        return shaped_data, ptv, oar, coarse_dose, directionmap


    def drawline(self, image, degreelist, edges, id):
        directionmap = np.zeros((9,128,128))
        for idx, degree in enumerate(degreelist):
            k = math.tan(math.radians(degree))
            b = []
            bound = np.where(edges != 0)
            for i in range(bound[0].size):
                m = bound[0][i]; n = bound[1][i]      
                b.append(k*m + n)
            min_loc = b.index(min(b)); x0 = bound[0][min_loc]; y0 = bound[1][min_loc]
            max_loc = b.index(max(b)); x02 = bound[0][max_loc]; y02 = bound[1][max_loc]
            thick = 5 
            for x in range(image.shape[1]):
                y_min = int(-k*(x - x0) +y0 - thick*math.sqrt(k**2+1))
                y_max = int(-k*(x - x02) + y02 + thick*math.sqrt(k**2+1))
                y = (k for k in range(y_min, y_max+1) if k >=0 and k < image.shape[1])
                for kk in y:
                    directionmap[idx,x,kk] = 1
        return directionmap

    def cal_distance_ct_ptv(self, ptv, ct, ptv_bound, coordinate):
        bounds = np.array(coordinate)
        # print('bounds:',bounds.shape)
        tree = neighbors.KDTree(bounds, leaf_size=bounds.shape[0] / 2)

        # bounds_1 = np.array(coordinate_1)
        # tree_1 = neighbors.KDTree(bounds_1, leaf_size=bounds_1.shape[0] / 2)
        # calculate distance between ct and ptv
        ct_nonzero = np.nonzero(ct)
        m = []
        for i in range(ct_nonzero[1].size):
            m.append([ct_nonzero[1][i], ct_nonzero[2][i], ct_nonzero[3][i]])
        '''
        m = np.mgrid[0:128, 0:128, 0:128].reshape(3, -1).transpose()
        '''

        dist_ct_ptv, ind = tree.query(m, k=1)
        # dist_ct_oar, ind_1 = tree_1.query(m, k=1)

        dist_ct_ptv = (dist_ct_ptv - min(dist_ct_ptv)) / (max(dist_ct_ptv) - min(dist_ct_ptv))

        # generate distance numpy
        dis_ct_ptv = np.zeros((128, 128, 128, 1), dtype=np.float32)
        c = 0
        for i in m:
            # if oar[0, i[0], i[1], i[2]] != 0:
            #     dis_ct_ptv[i[0], i[1], i[2]] = -dist_ct_ptv[c]
            if ptv[0, i[0], i[1], i[2]] != 0:
                dis_ct_ptv[i[0], i[1], i[2]] = 1
            else:
                dis_ct_ptv[i[0], i[1], i[2]] = (1-dist_ct_ptv[c])*0.7
            c += 1
        dis_ct_ptv = dis_ct_ptv.transpose((3,0,1,2))
        return np.clip(dis_ct_ptv, 0,1)

    def resample_image(self, in_arr, spacing, rotate, linear_interpolate, flip_x, flip_z):
        out_arr = np.zeros_like(in_arr)
        size = in_arr.shape[1:4]
        center = size * spacing * 0.5
        imagetype = itk.Image[itk.F, 3]

        if rotate == 0:
            transform = itk.IdentityTransform[itk.D, 3].New()
        else:
            transform = itk.Euler3DTransform[itk.D].New()
            transform.SetCenter((center[2], center[1], center[0]))
            transform.SetRotation(0, 0, rotate * 3.141592654 / 180)

        for i in range(in_arr.shape[0]):
            image = itk.GetImageFromArray(in_arr[i, :])
            image.SetSpacing((spacing[2], spacing[1], spacing[0]))
            resampler = itk.ResampleImageFilter[imagetype, imagetype].New()
            resampler.SetInput(image)
            resampler.SetSize(image.GetBufferedRegion().GetSize())
            resampler.SetOutputSpacing(image.GetSpacing())
            resampler.SetOutputOrigin(image.GetOrigin())
            resampler.SetTransform(transform)
            if linear_interpolate:
                resampler.SetInterpolator(itk.LinearInterpolateImageFunction[imagetype, itk.D].New())
            else:
                resampler.SetInterpolator(itk.NearestNeighborInterpolateImageFunction[imagetype, itk.D].New())
            resampler.SetDefaultPixelValue(0)
            resampler.Update()
            image = resampler.GetOutput()
            image_array = itk.GetArrayFromImage(image)
            out_arr[i, :] = image_array

        if flip_x:
            out_arr = np.ascontiguousarray(np.flip(out_arr, axis=3))
        if flip_z:
            out_arr = np.ascontiguousarray(np.flip(out_arr, axis=1))

        return out_arr

    def rescale_image(self, in_arr, spacing, rescale_factor, linear_interpolate=True, flip=False):
        out_arr = np.zeros_like(in_arr)
        imagetype = itk.Image[itk.F, 3]
        size = np.zeros(3, dtype=np.int32)
        size[0] = in_arr.shape[3]
        size[1] = in_arr.shape[2]
        size[2] = in_arr.shape[1]
        src_spacing = np.zeros(3, dtype=np.float32)
        src_spacing[0] = spacing[2]
        src_spacing[1] = spacing[1]
        src_spacing[2] = spacing[0]
        src_origin = np.zeros(3, dtype=np.float32)
        dst_spacing = rescale_factor * src_spacing
        dst_origin = src_origin + 0.5 * src_spacing * size - 0.5 * dst_spacing * size
        for i in range(in_arr.shape[0]):
            src_image = itk.GetImageFromArray(in_arr[i, :])
            src_image.SetOrigin((float(src_origin[0]), float(src_origin[1]), float(src_origin[2])))
            src_image.SetSpacing((float(src_spacing[0]), float(src_spacing[1]), float(src_spacing[2])))
            resampler = itk.ResampleImageFilter[imagetype, imagetype].New()  # 图像重采样
            resampler.SetInput(src_image)
            resampler.SetSize((int(size[0]), int(size[1]), int(size[2])))
            resampler.SetOutputSpacing((float(dst_spacing[0]), float(dst_spacing[1]), float(dst_spacing[2])))
            resampler.SetOutputOrigin((float(dst_origin[0]), float(dst_origin[1]), float(dst_origin[2])))
            resampler.SetTransform(itk.IdentityTransform[itk.D, 3].New())
            if linear_interpolate:
                resampler.SetInterpolator(itk.LinearInterpolateImageFunction[imagetype, itk.D].New())
            else:
                resampler.SetInterpolator(itk.NearestNeighborInterpolateImageFunction[imagetype, itk.D].New())
            resampler.SetDefaultPixelValue(0)
            resampler.Update()
            dst_image = resampler.GetOutput()
            dst_image = itk.GetArrayFromImage(dst_image)
            # dst_image = itk.GetImageFromArray(dst_image)
            # dst_image.SetOrigin(src_image.GetOrigin())
            # dst_image.SetSpacing(src_image.GetSpacing())
            out_arr[i, :] = dst_image

        if flip:
            np.flip(out_arr, axis=3)

        return out_arr


# if __name__ == '__main__':
#     import time
#     print("Start")
#     starttime = time.time()

#     dataset_train = KBPDataset(get_paths('/public/bme/home/v-tenglin/BDCLDosePrediction/Data/test'), flipped=False, rotate=False, noise=False, deformation=False, ct_range=[-400, 2400])
#     data_loader_train = data.DataLoader(
#         dataset=dataset_train, batch_size=1, shuffle=False, pin_memory=True, drop_last=False, num_workers=0)
    
#     for i, data in enumerate(data_loader_train):
#         print(data['patient_id'])
    
#     endtime = time.time()   

#     print("\nthis is the size of the dataset", len(dataset_train)) 
#     dtime = endtime - starttime

#     print("time: %.8s s" % dtime)  #显示到微秒

