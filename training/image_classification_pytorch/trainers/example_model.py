# coding=utf-8
from curses import def_prog_mode
import os
import math
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torchvision.models
from PIL import Image

from trainers.base_model import BaseModel
from nets.net_interface import NetModule
import matplotlib.pyplot as plt

import numpy as np

class ExampleModel(BaseModel):
    def __init__(self, config, time, suffix):
        super(ExampleModel, self).__init__(config, time, suffix)
        self.config = config
        # self.interface = NetModule(self.config['model_module_name'], self.config['model_net_name'])
        self.create_model()


    def create_model(self):
        # self.net = self.interface.create_model(num_classes=self.config['num_classes'])
        import torchvision.models as models

        print('loading model ', self.config['model_net_name'])

        if self.config['model_net_name'] == 'mobilenet_v3_large':
            self.feature_dimension = 960
            self.classifier = nn.Linear(self.feature_dimension, 2).cuda()
            self.net = models.mobilenet_v3_large(pretrained=False)
            print('cwd ', os.getcwd())
            state_dict = torch.load('./image_classification_pytorch/checkpoints/mobnet_v3_large_imgnet.pth')
            print('loading imagenet weights for mobilenet', self.net.load_state_dict(state_dict))
            # self.net.classifier = nn.Linear(self.feature_dimension, 1)
            self.net.classifier = nn.Identity()
        elif self.config['model_net_name'] == 'resnet18':
            self.feature_dimension = 512
            self.classifier = nn.Linear(self.feature_dimension, 2).cuda()
            self.net = models.resnet18(pretrained=False)
            print('cwd ', os.getcwd())
            state_dict = torch.load('./image_classification_pytorch/checkpoints/resnet18_imgnet.pth')
            print('loading imagenet weights for resnet', self.net.load_state_dict(state_dict))
            # self.net.classifier = nn.Linear(self.feature_dimension, 1)
            self.net.fc = nn.Identity()



        #self.net = torchvision.models.resnet18(pretrained=True)
        #print("resnet loaded")
        #self.net.fc = nn.Linear(self.net.fc.in_features, self.config['num_classes'])
        if torch.cuda.is_available():
            self.net.cuda()


    def load(self):
        # train_mode: 0:from scratch, 1:finetuning, 2:update
        # if not update all parameters:
        # for param in list(self.net.parameters())[:-1]:    # only update parameters of last layer
        #    param.requires_grad = False
        train_mode = self.config['train_mode']
    
        if train_mode == 'fromscratch':
            if torch.cuda.device_count() > 1:
                self.net = nn.DataParallel(self.net)
            if torch.cuda.is_available():
                self.net.cuda()
            print('from scratch...')

        elif train_mode == 'finetune':
            self._load()
            if torch.cuda.device_count() > 1:
                self.net = nn.DataParallel(self.net,device_ids=range(torch.cuda.device_count()))
            if torch.cuda.is_available():
                self.net.cuda()
            print('finetuning...')

        elif train_mode == 'update':
            self._load()
            print('updating...')

        elif train_mode == 'pre_trained':  # for test
            self._load()
            print('load_pretrained_model')

        else:
            ValueError('train_mode is error...')


    def _load(self):
        _state_dict = torch.load(os.path.join(self.config['pretrained_path'], self.config['pretrained_file']),
                              map_location=None) 
                                
        
        #_state_dict = torch.load(os.path.join(self.config['save_path'], self.config['save_name']),
        #                      map_location=None) # loading best model
        
        # # for multi-gpus
        # state_dict = OrderedDict()
        # for item, value in _state_dict.items():
        #     if 'module' in item.split('.')[0]:
        #         name = '.'.join(item.split('.')[1:])
        #     else:
        #         name = item
        #     state_dict[name] = value
        # for handling in case of different models compared to the saved pretrain-weight
        model_dict = self.net.state_dict()
        # same = {k: v for k, v in state_dict.items() if \
        #         (k in model_dict and model_dict[k].size() == v.size())}  # or (k not in state_dict)}
        diff = {k: v for k, v in _state_dict.items() if \
                (k in model_dict and model_dict[k].size() != v.size()) or (k not in model_dict)}
        print('diff: ', [i for i, v in diff.items()])
        # model_dict.update(same)
        model_dict = self.net.state_dict()
        # print('state_dcit')
        # print(_state_dict["state_dict"].keys())
        # print('model stet dict')
        # print(self.net.state_dict().keys())
        # print('model dict before ', model_dict[list(model_dict.keys())[0]])
        # _state_dict["state_dict"] = {k[len('module.'):]:v for k,v in _state_dict["state_dict"].items()}
        # print(' after state_dcit')

        # print(_state_dict["state_dict"].keys())
        # asd
        model_dict.update(_state_dict["state_dict"])

        print('updated mode_dict', model_dict[list(model_dict.keys())[0]])
        print(self.net.load_state_dict(model_dict))
        print('model loaded')
    def load_test(self):
        _state_dict = torch.load(self.config['test'],
                              map_location=None) 

        # print('model state dict')
        # print(self.state_dict().keys())


        # print('pretrained state dict')
        # print(_state_dict.keys())
        print(self.load_state_dict(_state_dict))
        # asd
        print('model loaded')

    def forward(self, images):
        b, n, c, h , w = images.shape
        # print('before reshape:', images.shape)
    
        images = images.reshape((-1, *images.shape[2:]))
        
        # print('before model after rehshape reshape:', images.shape)
        outputs = self.net(images)

        # ou

        # print('after model ', outputs.shape)


        
        outputs = outputs.reshape(b, n, self.feature_dimension)

        # print('outputs ', outputs.shape)

        outputs = outputs.amax(axis=1)
        logits = self.classifier(outputs).squeeze()
        # print('logits ', logits.shape)

        # asd

        # out_ = outputs.squeeze()


        # print('before max out_ ', out_.shape)
        # logits = torch.max(out_, dim=1).values
        # print('logits ', [logits.min(), logits.max(), logits.shape])
        # asd

        return logits




class ExampleModel_3d(BaseModel):
    def __init__(self, config):
        super(ExampleModel_3d, self).__init__(config)
        self.config = config
        self.interface = NetModule(self.config['model_module_name'], self.config['model_net_name'])
        self.create_model()


    def create_model(self):
        self.net = self.interface.create_model(num_classes=self.config['num_classes'])
        #self.net = torchvision.models.resnet18(pretrained=True)
        #print("resnet loaded")
        #self.net.fc = nn.Linear(self.net.fc.in_features, self.config['num_classes'])
        if torch.cuda.is_available():
            self.net.cuda()


    def load(self):
        # train_mode: 0:from scratch, 1:finetuning, 2:update
        # if not update all parameters:
        # for param in list(self.net.parameters())[:-1]:    # only update parameters of last layer
        #    param.requires_grad = False
        train_mode = self.config['train_mode']
    
        if train_mode == 'fromscratch':
            print('loaded')
            if torch.cuda.device_count() > 1:
                self.net = nn.DataParallel(self.net)
            if torch.cuda.is_available():
                self.net.cuda()
            print('from scratch...')

        elif train_mode == 'finetune':
            self._load()
            if torch.cuda.device_count() > 1:
                self.net = nn.DataParallel(self.net,device_ids=range(torch.cuda.device_count()))
            if torch.cuda.is_available():
                self.net.cuda()
            print('finetuning...')

        elif train_mode == 'update':
            self._load()
            print('updating...')

        elif train_mode == 'pre_trained':  # for test
            self._load()
            print('load_pretrained_model')

        else:
            ValueError('train_mode is error...')


    def _load(self):
        _state_dict = torch.load(os.path.join(self.config['pretrained_path'], self.config['pretrained_file']),
                              map_location=None) 
                                
        
        #_state_dict = torch.load(os.path.join(self.config['save_path'], self.config['save_name']),
        #                      map_location=None) # loading best model
        
        # # for multi-gpus
        # state_dict = OrderedDict()
        # for item, value in _state_dict.items():
        #     if 'module' in item.split('.')[0]:
        #         name = '.'.join(item.split('.')[1:])
        #     else:
        #         name = item
        #     state_dict[name] = value
        # for handling in case of different models compared to the saved pretrain-weight
        model_dict = self.net.state_dict()
        # same = {k: v for k, v in state_dict.items() if \
        #         (k in model_dict and model_dict[k].size() == v.size())}  # or (k not in state_dict)}
        diff = {k: v for k, v in _state_dict.items() if \
                (k in model_dict and model_dict[k].size() != v.size()) or (k not in model_dict)}
        print('diff: ', [i for i, v in diff.items()])
        # model_dict.update(same)
        model_dict = self.net.state_dict()
        # print('state_dcit')
        # print(_state_dict["state_dict"].keys())
        # print('model stet dict')
        # print(self.net.state_dict().keys())
        # print('model dict before ', model_dict[list(model_dict.keys())[0]])
        # _state_dict["state_dict"] = {k[len('module.'):]:v for k,v in _state_dict["state_dict"].items()}
        # print(' after state_dcit')

        # print(_state_dict["state_dict"].keys())
        # asd
        model_dict.update(_state_dict["state_dict"])

        # print('updated mode_dict', model_dict[list(model_dict.keys())[0]])
        # print(self.net.load_state_dict(model_dict))
        print('model loaded')


    def forward(self, images):
        b, n, c, h , w = images.shape
        # print('before model:', images.shape)
        images = images.permute(0,2,1,3,4)
        # print('permutation:', images.shape)
    
       #images = images.reshape((-1, *images.shape[2:]))
        
        #print('before model reshape:', images.shape)
        outputs = self.net(images)
       
        
        #print('labels: ', out_)
        # print('after model shape:', outputs.shape)
        
        #out_ = out_.reshape(b, n, self.config['num_classes'])
        # out_ = out_.reshape(b, n)
        #print(out_)
        logits = outputs.squeeze()
        #print('squeeze', logits.shape)
    
        # logits = torch.max(out_, dim=1).values
        # #print('pred: ', logits)
        # #print('preds', logits)
        # print( "size", logits.shape)
        #asd
        return logits