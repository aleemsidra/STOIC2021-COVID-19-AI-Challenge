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

from multiprocessing import Pool
import functools

def sub_samples(path, i):

    # print('path + i ', path + i)
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

def preprocess_single(i, args):
    path = args[0]
    root_dir = args[1]
    sub_path = root_dir
    # print('i', i)
    # print('path ', root_dir)
    # print('root_dir ', path)
    if not os.path.exists(os.path.join(sub_path , i.split('.')[0])):
        os.mkdir(os.path.join(sub_path , i.split('.')[0]))

    sub_sample = sub_samples(path, i)

    image = window(sub_sample, 1500, -600)
    for j in range(image.shape[0]):
        cv2.imwrite(os.path.join(sub_path , i.split('.')[0], str(j) + '.png'), image[j,:,:]*255)


def preprocess_dir(mha_dir, scratch_dir):


    img_list = os.listdir(mha_dir)
    # sub_path = os.path.join(root_dir, "scratch")


    img_list = [i for i in img_list if '.mha' in i]

    print('total images to be preprocessed ', len(img_list))
    # asd
    with Pool(16) as p:
        r = list(tqdm(p.map(functools.partial(preprocess_single, args=(mha_dir, scratch_dir)), img_list), total=len(img_list)) )

