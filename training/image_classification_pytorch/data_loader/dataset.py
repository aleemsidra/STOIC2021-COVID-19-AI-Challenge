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
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2


class PyTorchDataset(Dataset):
    def __init__(self, txt, config, transform=None, loader = None,
                 target_transform=None,  is_train_set=True):
        self.config = config
        imgs = []
        with open(txt,'r') as f:
            for line in f:
                line = line.strip('\n\r').strip('\n').strip('\r')
                words = line.split(self.config['file_label_separator'])
                # single label here so we use int(words[1])
                imgs.append((words[0], int(words[1])))

        self.DataProcessor = DataProcessor(self.config)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.is_train_set = is_train_set


    def __getitem__(self, index):
        fn, label = self.imgs[index]
        _root_dir = self.config['train_data_root_dir'] if self.is_train_set else self.config['val_data_root_dir']
        image = self.self_defined_loader(os.path.join(_root_dir, fn))
        if self.transform is not None:
            image = self.transform(image)

        return image, label


    def __len__(self):
        return len(self.imgs)


    def self_defined_loader(self, filename):
        image = self.DataProcessor.image_loader(filename)
        image = self.DataProcessor.image_resize(image)
        if self.is_train_set and self.config['data_aug']:
            image = self.DataProcessor.data_aug(image)
        image = self.DataProcessor.input_norm(image)
        return image

class chest_ct(Dataset):

    def __init__(self, csv_file, root_dir, transform, loader = pil_loader):

        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.loader = loader
        self.labels_severity = list(self.labels.iloc[:, 2])
        self.column_name = self.labels.columns[2]
        #print('self.column_name', self.column_name)
        
        # print(self.labels_severity) 
        # print(len(self.labels_severity))
  

    def __len__(self):
        
        return len(self.labels)  
    
    def __getitem__(self, idx):
        
        output_list =[]
        img_name = os.path.join(self.root_dir,
                                str(self.labels.iloc[idx, 0]))
        label = self.labels.iloc[idx, 2]
        label = torch.from_numpy(np.asarray(label))

        label2 = self.labels.iloc[idx, 1]
        label2 = torch.from_numpy(np.asarray(label2))
   
        for img in os.listdir(img_name):
            image = cv2.imread(os.path.join(img_name,img)) 
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # image = self.transform(image)
            # print(image.shape)
            # asd
            output_list.append(image)
        # print(['output ', len(output_list), output_list[0].shape])
        output_list = self.apply_transform(output_list)
        # print(['output after ', len(output_list), output_list[0].shape])
        # asd
        image = torch.stack(output_list)
        # print(['image ', image.shape ])
        return image, (label, label2)
    
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
    train_data_file = config['train_data_file']
    val_data_file = config['val_data_file']
    train_data_root_dir = config['train_data_root_dir']
    val_data_root_dir = config['val_data_root_dir']
    img_width = config['img_width']
    batch_size = config['batch_size']
    num_workers =config['dataloader_workers']
 


    if not os.path.isfile(train_data_file):
        raise ValueError('train_data_file is not existed')
    # if not os.path.isfile(test_data_file):
    #     raise ValueError('val_data_file is not existed')


    # train_transform = transforms.Compose([
    # transforms.Resize(img_width),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(192),
    # transforms.ToTensor(), 
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # train_transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize(img_width),
    #     #transforms.RandomResizedCrop(224),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     #transforms.RandomAffine(translate=(0.1, 0.3), scale=(0.5, 0.75), degrees=15),
    #     transforms.Normalize((0.5, ), (0.5, ))
    #         ])


    # train_transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize(img_width),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomResizedCrop(224),
    #     # transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     # transforms.RandomAffine(translate=(0.1, 0.3), scale=(0.5, 0.75), degrees=15),
    #     transforms.Normalize((0.5, ), (0.5, ))
    #         ])

    # train_transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize(img_width),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop(224),
    #     # transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     # transforms.RandomAffine(translate=(0.1, 0.3), scale=(0.5, 0.75), degrees=15),
    #     transforms.Normalize((0.5, ), (0.5, ))
    #         ])



    if config['train_augs']=='default':
        train_transform = [
            albu.Resize(img_width, img_width),
            albu.HorizontalFlip(p=0.5),
            albu.RandomCrop(224, 224),
            albu.RandomGamma(),
            albu.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.0),
            albu.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ]
    elif config['train_augs']=='strong':
        train_transform = [
        albu.geometric.SafeRotate(limit=30, border_mode=0, p=0.7),
        albu.Resize(img_width+32, img_width+32),
        # albu.Resize(img_width, img_width),
        albu.HorizontalFlip(p=0.5),
        albu.RandomCrop(224, 224),
        albu.MedianBlur(),
        albu.RandomGamma(),
        albu.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.0),
    #     albu.RandomToneCurve(),
        # albu.MultiplicativeNoise(),

        albu.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ]

    print('train transforms are ', train_transform)

    train_transform = albu.ReplayCompose(train_transform)



    # valid_transform = transforms.Compose([
        
    #     transforms.ToPILImage(),
    #     transforms.Resize(img_width),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, ), (0.5, ))


    #     ])

    valid_transform = [
        albu.Resize(img_width, img_width),
        albu.CenterCrop(224, 224),
        albu.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ]

    valid_transform = albu.ReplayCompose(valid_transform)



    train_data = chest_ct(csv_file = train_data_file,
                    root_dir = train_data_root_dir,
                    transform = train_transform)
    
 
    val_data = chest_ct(csv_file = val_data_file,
                    root_dir = val_data_root_dir,
                    transform = valid_transform)
    #print('val', len(val_data))

    # test_data = chest_ct(csv_file = test_data_file,
    #                 root_dir = test_data_root_dir,
    #                 transform = valid_transform)
    
    #print('test', len(test_data))
    
    # train_data = PyTorchDataset(txt=train_data_file,config=config,
    #                        transform=transforms.ToTensor(), is_train_set=True)
    # test_data = PyTorchDataset(txt=test_data_file,config=config,
    #                             transform=transforms.ToTensor(), is_train_set=False)
    # print('shape:', train_data[0][0][0].shape)
    # print(train_data[0][0][0].min(), train_data[0][0][0].max())
    #plt.imsave('/home/sidra/Documents/final.png', train_data[0][0][10].permute(1,2,0).numpy().squeeze(), cmap = 'gray') 
    weights = make_weights_for_balanced_classes(train_data.labels_severity, 2)                                                                
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     
    # asd

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size,
                              num_workers=num_workers, sampler=sampler, drop_last=True)
    

    
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    # test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
    #                          num_workers=num_workers)


    return train_loader, val_loader

def make_weights_for_balanced_classes(labels, nclasses):                        
    count = [0] * nclasses                                                      
    for item in labels:                                                         
        count[item] += 1 
                                                        
    weight_per_class = [0.] * nclasses  
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])
    # print('weight_per_class', weight_per_class)                                    
                                 
    weight = [0] * len(labels)                                              
    for idx, val in enumerate(labels):                                          
        weight[idx] = weight_per_class[val]
    # print('weight', weight)
    # asd                                  
    return weight
