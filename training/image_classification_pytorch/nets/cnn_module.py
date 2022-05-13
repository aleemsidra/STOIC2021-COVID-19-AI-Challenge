import torch.nn as nn
import torch
import torchvision.transforms as tfms
import sys
import torch.nn.functional as F


# Basic ConvNet with Pooling layer
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
class ConvNet(nn.Module):
    def __init__(self, x_dim, hid_dim, z_dim,  max_pool, num_classes, resize=True):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.resize = resize
        self.max_pool = max_pool
        self.fc = nn.Linear(1*1*64, num_classes) 
   
    def forward(self, x):
      

        x = self.encoder(x)
        #print('conv block: ', x.shape)
        
        if self.max_pool is not None:
            if self.max_pool == 'max_pool':
                #print('using max_pool')
                x = nn.AdaptiveMaxPool2d(1)(x)
            elif self.max_pool == 'avg_pool':
                x = nn.AdaptiveAvgPool2d(1)(x)
        #print('after pooling: ', x.shape)
        if self.resize:
            x = x.view(x.size(0), -1)
        #print('x', x.shape)

        x = self.fc(x)
        #print('fc', x.shape)
        return x



def cnn(**kwargs):
    
    num_classes = kwargs.get('num_classes', 1000)
    model = ConvNet(1, 64,64, 'max_pool', num_classes,'True')
    return model
