# coding=utf-8

import math
import os
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

class BaseModel(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.config = config
        # self.time = time
        # self.suffix = suffix

    # save function thet save the checkpoint in the path defined in configfile
    def save(self, auc):
        """
        implement the logic of saving model
        """
        print("Saving model...")
        save_path = self.config['save_path']

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print("path :" , save_path)
     


        try:
            if not os.path.exists(os.path.join (save_path, self.config['model_net_name'])):
                self.save_dir = os.path.join(save_path, self.config['model_net_name'] + "_"+self.suffix +"_" +self.time)
                os.makedirs(self.save_dir)
        except:
             pass




        #save_name = os.path.join(save_path,self.config['save_name'])
        if auc == 'True':  
            save_name = os.path.join(self.save_dir,self.config['auc_save_name'])
        else:
            save_name = os.path.join(self.save_dir,self.config['acc_save_name'])
           
               
        
        state_dict = OrderedDict()
        for item, value in self.net.state_dict().items():
            if 'module' in item.split('.')[0]:
                name = '.'.join(item.split('.')[1:])
            else:
                name = item
            state_dict[name] = value
        torch.save(state_dict, save_name)
        print("Model saved: ", save_name)
       
    # load lateset checkpoint from the experiment path defined in config_file
    def load(self):
        """
        implement the logic of loading model
        """
        raise NotImplementedError


    def build_model(self):
        """
        implement the logic of model
        """
        raise NotImplementedError