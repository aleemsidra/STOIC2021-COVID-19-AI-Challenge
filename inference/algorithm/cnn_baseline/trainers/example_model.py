# coding=utf-8
from curses import def_prog_mode
import os

from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.models


from .base_model import BaseModel
from ..nets.net_interface import NetModule

class ExampleModel(BaseModel):
    def __init__(self, config):
        super(ExampleModel, self).__init__(config)
        self.config = config
        self.create_model()


    def create_model(self):
        import torchvision.models as models

        if self.config['model_net_name'] == 'mobilenet_v3_module':
            self.feature_dimension = 960
            self.classifier = nn.Linear(self.feature_dimension, 2).cuda()
            self.net = models.mobilenet_v3_large(pretrained=False)
            self.net.classifier = nn.Identity()
        elif self.config['model_net_name'] == 'resnet18':
            self.feature_dimension = 512
            self.classifier = nn.Linear(self.feature_dimension, 2).cuda()
            self.net = models.resnet18(pretrained=False)
            self.net.fc = nn.Identity()


        if torch.cuda.is_available():
            self.net.cuda()


    def load(self):
        _state_dict = torch.load(os.path.join(self.config['pretrained_path'], self.config['pretrained_file']),
                              map_location=None) 
                                
        print(self.load_state_dict(_state_dict))
        print('model loaded ', self.config['pretrained_file'])


    def forward(self, images):
        b, n, c, h , w = images.shape
    
        images = images.reshape((-1, *images.shape[2:]))
        
        outputs = self.net(images)

        outputs = outputs.reshape(b, n, self.feature_dimension)

        # outputs = outputs.amax(axis=1)
        outputs = torch.topk(outputs, 5, dim=1).values
        outputs = outputs.mean(1)

        
        logits = self.classifier(outputs).squeeze()

        return logits
