# coding=utf-8

import random
import numpy as np
import cv2
import scipy.misc as misc
import SimpleITK as sitk
import os
from data_loader.data_augmentation import DataAugmenters


##########################################################
# name:     DataProcessor
# breif:
#
# usage:
##########################################################
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.DataAugmenters = DataAugmenters(self.config)

    
    def sub_samples(path, i):
    
        input_image = sitk.ReadImage(path + i)
    #     print('input_image.shape ', input_image.GetSize())
        #slice = int(np.round(input_image.GetSize()[2]/33))
        slice = int(np.floor(input_image.GetSize()[2]/33))
        sub_sample = input_image[:, :, 0:input_image.GetDepth():slice]
        sub_sample = sitk.GetArrayFromImage(sub_sample) 
        sub_sample = sub_sample[:32, :, :]
    #     print('subsample.shape', sub_sample.shape)

        return sub_sample
        
        
    # windowing
    def window(img, w, l):
    
        x = l + w/2
        y = l - w/2
        img[img > x] = x
        img[img < y] = y
        chanel = (img - np.min(img))/(np.max(img) - np.min(img))
        
        return chanel
        
        
    
    def image_loader(self, filename, **kwargs):
        """
        load your image data
        :param filename: 
        :return: 
        """
        image = cv2.imread(filename)
        if image is None:
            raise ValueError('image data is none when cv2.imread!')

        # single window
        #path = "/home/sidra/Documents/STOIC/data/mha/"
        path = self.config['test_data_dir']
        img_list = os.listdir(path)  
        image_list =[]
        # img_list
        # sub_path = '/home/sidra/Documents/STOIC/data/single_window_sub_samples/'

        for i in tqdm(img_list):
            
            # if not os.path.exists(sub_path + i.split('.')[0]):
            #     os.mkdir(sub_path + i.split('.')[0])
            
            sub_sample = sub_samples(path, i)   


            image = window(sub_sample, 1500, -600)

            
            # cv2.imwrite(os.path.join(sub_path + (i.split('.')[0]), str(j) + '.png'), ch_1[j,:,:]*255)

        # for j in range(ch_1.shape[0]):
        #     cv2.imwrite(os.path.join(sub_path + (i.split('.')[0]), str(j) + '.png'), ch_1[j,:,:]*255)
       
            return image


    def image_resize(self, image, **kwargs):
        """
        resize your image data
        :param image: 
        :param kwargs: 
        :return: 
        """
        _size = (self.config['img_width'], self.config['img_height'])
        _resize_image = cv2.resize(image, _size)
        return _resize_image[:,:,::-1]  # bgr2rgb

    def input_norm(self, image, **kwargs):
        """
        normalize your image data
        :param image: 
        :return: 
        """
        return ((image - 127) * 0.0078125).astype(np.float32) # 1/128


    def data_aug(self, image, **kwargs):
        """
        augment your image data with DataAugmenters
        :param image: 
        :return: 
        """
        return self.DataAugmenters.run(image, **kwargs)


if __name__ == '__main__':

    print('done!')
