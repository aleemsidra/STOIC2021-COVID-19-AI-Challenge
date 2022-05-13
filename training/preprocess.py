import sys 
import numpy as np
import SimpleITK as sitk
import os
import cv2
import pandas as pd
import random
import shutil
from PIL import Image 
from tqdm import tqdm

def sub_samples(path, i):
    
    input_image = sitk.ReadImage(path + i)
#     print('input_image.shape ', input_image.GetSize())

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



def process_dir():


path = "/home/sidra/Documents/STOIC/stoic_2022_final_phase/data/mha/"
root_dir = "/home/sidra/Documents/STOIC/stoic_2022_final_phase/"

img_list = os.listdir(path)  
sub_path = os.path.join(root_dir, "scratch")

j=0
for i in tqdm(img_list):
 if not os.path.exists(os.path.join(sub_path , i.split('.')[0])):
    os.mkdir(os.path.join(sub_path , i.split('.')[0]))

 sub_sample = sub_samples(path, i)   
 j=j+1

 image = window(sub_sample, 1500, -600)
 for j in range(image.shape[0]):
    cv2.imwrite(os.path.join(sub_path , i.split('.')[0], str(j) + '.png'), image[j,:,:]*255)