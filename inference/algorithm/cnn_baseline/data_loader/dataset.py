# -*- coding: utf-8 -*-

from dataclasses import asdict
from fnmatch import translate
import os
import cv2
import copy
import numpy as np
import torch
import pandas as pd
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from data_loader.data_processor import DataProcessor
from torchvision.datasets.folder import pil_loader
import shutil
import matplotlib.pyplot as plt 
import SimpleITK as sitk
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2


        
class chest_ct(Dataset):

    def __init__(self, csv_file, root_dir, transform, loader = pil_loader):

        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.loader = loader
        self.labels_severity = list(self.labels.iloc[:, 2])
        self.column_name = self.labels.columns[2]

  

    def __len__(self):
        
        return len(self.labels)  
    
    def __getitem__(self, idx):
        
        output_list =[]
        w = 1500
        l= -600
        img_name = os.path.join(self.root_dir,
                                str(self.labels.iloc[idx, 0]))
        label = self.labels.iloc[idx, 2]
        label = torch.from_numpy(np.asarray(label))
        
        #sub sample
        input_image = sitk.ReadImage(img_name + ".mha")
        # print('input_image.shape ', input_image.GetSize())
        #slice = int(np.round(input_image.GetSize()[2]/33))
        slice = int(np.floor(input_image.GetSize()[2]/33))
        sub_sample = input_image[:, :, 0:input_image.GetDepth():slice]
        sub_sample = sitk.GetArrayFromImage(sub_sample) 
        sub_sample = sub_sample[:32, :, :]
        # print('subsampel type', type(sub_sample))    
        # print('subsample.shape', sub_sample.shape)
        # windowing
        x = l + w/2
        y = l - w/2
        sub_sample[sub_sample > x] = x
        sub_sample[sub_sample < y] = y
        chanel = (sub_sample - np.min(sub_sample))/(np.max(sub_sample) - np.min(sub_sample))
        print(chanel.shape)
        for img in range(chanel.shape[0]):
            img
            image =np.asarray(chanel[img,:,:]).astype(np.float32)

            output_list.append(image)

        output_list = self.apply_transform(output_list)
        image = torch.stack(output_list)
        return image, label
    def apply_transform(self, images):
        if self.transform is None:
            return images
        if isinstance(self.transform, albu.ReplayCompose):
            results = []
            result = self.transform(image=images[0])
            results.append(result['image'])
            replay = result['replay']
            for image in images[1:]:
                results.append(self.transform.replay(replay, image=image)['image'])
            return results
        return [self.transform.apply(image=image)['image'] for image in images]


def get_data_loader(config):
    """
    
    :param config: 
    :return: 
    """

    test_data_file = config['test_data_file']
    test_data_root_dir = config['test_data_root_dir']
    img_width = config['img_width']
    batch_size = config['batch_size']
    num_workers =config['dataloader_workers']
 


    if not os.path.isfile(test_data_file):
        raise ValueError('test_data_file is not existed')


    test_transform = [
        albu.Resize(img_width, img_width),
        albu.CenterCrop(224, 224),
        albu.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ]

    test_transform = albu.ReplayCompose(test_transform)

    test_data = chest_ct(csv_file = test_data_file,
                    root_dir = test_data_root_dir,
                    transform = test_transform)
    
    #print('test', len(test_data))

 
    # print('shape:', train_data[0][0][0].shape)
    # print(train_data[0][0][0].min(), train_data[0][0][0].max())
    # plt.imsave('/image_classification_pytorch/testing.png', test_data[0][0][16].permute(1,2,0).numpy().squeeze(), cmap = 'gray') 



    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)


    return test_loader


