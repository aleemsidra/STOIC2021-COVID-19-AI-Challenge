# coding=utf-8
from __future__ import print_function
import os, sys
import numpy as np
import logging
from datetime import datetime
import time
from utils import utils
from utils import logger_summarizer
import json
from utils.config import process_config
class ExampleLogger:
    """
    self.log_writer.info("log")
    self.log_writer.warning("warning log)
    self.log_writer.error("error log ")

    try:
        main()
    except KeyboardInterrupt:
        sys.stdout.flush()
    """
    def __init__(self, config, time, suffix):
        self.config = config
        self.time = time
        self.suffix = suffix
        self.log_writer = self.init()
        self.log_printer = DefinedPrinter()
        if self.config['is_tensorboard']:
            self.summarizer = logger_summarizer.Logger(self.config, self.model_dir)
        self.log_info = {}


    def init(self):
        """
        initial
        :return: 
        """
        log_writer = logging.getLogger(__name__)
        log_writer.setLevel(logging.INFO)
        self.log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        if not os.path.exists(os.path.join (self.log_dir, self.config['model_net_name'])):
             self.model_dir = os.path.join(self.log_dir, self.config['model_net_name'] + "_" +self.suffix+"_"+self.time)
             os.makedirs(self.model_dir)

       # saving config file
        with open(os.path.join(self.model_dir, 'config.json'), 'w') as fp:  
            json.dump(self.config, fp)


        handler = logging.FileHandler(os.path.join(self.model_dir, 'alg_training.log'),encoding='utf-8')
       
        handler.setLevel(logging.DEBUG)
        logging_format = logging.Formatter("@%(asctime)s [%(filename)s -- %(funcName)s]%(lineno)s : %(levelname)s - %(message)s", datefmt='%Y-%m-%d %A %H:%M:%S')
        handler.setFormatter(logging_format)
        log_writer.addHandler(handler)
        return log_writer


    def write_info_to_logger(self, variable_dict):
        """
        print
        :param variable_dict: 
        :return: 
        """
        if variable_dict is not None:
            for tag, value in variable_dict.items():
                self.log_info[tag] = value


    def write(self):
        """
        log writing
        :return: 
        """
        _info = 'epoch: %d, lr: %f, eval_train: %f, train_auc: %f, eval_validate: %f,  val_auc: %f, train_avg_loss: %f, validate_avg_loss: %f, gpu_index: %s, net: %s' % (
        self.log_info['epoch'],self.log_info['lr'], self.log_info['train_acc'], self.log_info['train_auc'], self.log_info['validate_acc'], 
        self.log_info['val_auc'], self.log_info['train_avg_loss'], self.log_info['validate_avg_loss'],
        self.log_info['gpus_index'], self.log_info['net_name'])

        self.log_writer.info(_info)
        sys.stdout.flush()


    def write_warning(self, warning_dict):
        """
        warninginfo writing
        :return: 
        """
        _info = 'epoch: %d, lr: %f, loss: %f'%(warning_dict['epoch'],warning_dict['lr'], warning_dict['loss'])
        self.log_writer.warning(_info)
        sys.stdout.flush()

    def clear(self):
        """
        clear log_info
        :return: 
        """
        self.log_info = {}


    def close(self):
        if self.config['is_tensorboard']:
            self.summarizer.train_summary_writer.close()
            self.summarizer.validate_summary_writer.close()



class DefinedPrinter:
    """
    Printer
    """

    def init_case_print(self, loss_start, eval_start_train, eval_start_val):
        """
        print when init
        :param loss_start: 
        :param eval_start_train: 
        :param eval_start_val: 
        :return: 
        """
        log = "\nInitial Situation:\n" + \
              "Loss= \033[1;32m" + "{:.6f}".format(loss_start) + "\033[0m, " + \
              "Training EVAL= \033[1;36m" + "{:.5f}".format(eval_start_train * 100) + "%\033[0m , " + \
              "Validating EVAL= \033[0;31m" + "{:.5f}".format(eval_start_val * 100) + '%\033[0m'
        print('\n\r', log)
        print('---------------------------------------------------------------------------')


    def iter_case_print(self, epoch, eval_train, eval_validate, limit, iteration, loss, lr, train_auc, val_auc):
        """
        print per batch 
        :param epoch: 
        :param eval_train: 
        :param eval_validate: 
        :param limit: 
        :param iteration: 
        :param loss: 
        :param lr: 
        :param global_step: 
        :return: 
        """

        log = "Epoch \033[1;33m" + str(epoch) + "\033[0m, " + \
              "Iter \033[1;33m" + str(iteration) + '/' + str(limit) + "\033[0m, " + \
              "Loss \033[1;32m" + "{:.6f}".format(loss) + "\033[0m, " + \
              "lr \033[1;37;45m" + "{:.6f}".format(lr) + "\033[0m, " + \
              "Training EVAL \033[1;36m" + "{:.5f}".format(eval_train * 100) + "%\033[0m, " + \
              "Training AUC \033[1;36m" + "{:.5f}".format(train_auc * 100) + "%\033[0m, " + \
              "Validating EVAL \033[1;31m" + "{:.5f}".format(eval_validate * 100) + "%\033[0m, " + \
              "Validating AUC \033[1;31m" + "{:.5f}".format(val_auc * 100) + "%\033[0m, "
        print(log)

    def epoch_case_print(self, epoch, eval_train, eval_validate, loss_train, loss_validate, train_auc, val_auc, fitTime):
        """
        print per epoch
        :param epoch: 
        :param eval_train: 
        :param eval_validate: 
        :param fitTime: 
        :return: 
        """
        log = "\nEpoch \033[1;36m" + str(epoch) + "\033[0m, " + \
              "Training EVAL \033[1;36m" + "{:.5f}".format(eval_train * 100) + "%\033[0m , " + \
              "Validating EVAL= \033[0;31m" + "{:.5f}".format(eval_validate * 100) + '%\033[0m , ' + \
              "\n\r" + \
              "Training avg_loss \033[1;32m" + "{:.5f}".format(loss_train) + "\033[0m, " + \
              "Training AUC \033[1;36m" + "{:.5f}".format(train_auc * 100) + "%\033[0m, " + \
              "Validating avg_loss \033[1;32m" + "{:.5f}".format(loss_validate) + "\033[0m, " + \
              "Validating AUC \033[1;31m" + "{:.5f}".format(val_auc * 100) + "%\033[0m, "+ \
              "epoch time " + str(fitTime) + ' ms' + '\n'
        print('\n\r', log, '\n\r')

