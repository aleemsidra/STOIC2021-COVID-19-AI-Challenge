# coding=utf-8
from __future__ import print_function
import os
import time
import argparse 
import torch
from torch.autograd import Variable
import torch
torch.cuda.empty_cache()
import wandb


#Initialize WandB 
#wandb.init(name='ct4', entity='sidra', mode = 'disabled')

class BaseTrainer:
    def __init__(self, model, test_loader,  config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        # self.train_loader = train_loader
        # self.val_loader = val_loader
        # self.time = time
        self.test_loader = test_loader
        # self.eval_train = 0.
        #self.eval_validate = 0
        # self.eval_test = 0 # new added
        # self.auc = 0
        # self.train_auc = 0 # new added
        self.eval_auc = 0 # new added
        self.optimizer = None
        self.loss = None
        

        #wandb.init(name=self.config['model_net_name'] + "_" +exp_suffix+"_"+ self.time, entity='sidra')
        # wandb.init(name=self.config['model_net_name'] , entity='sidra')

    def test(self):
    
     
        
        # for cur_epoch in range(1, total_epoch_num+1):
        #     self.cur_epoch = cur_epoch
        self.evaluate_epoch()
        self.eval_top1, self.validate_auc = self.eval_top1.value(), self.eval_auc.value()[0]
        print('accuracy:', self.eval_top1, "auc:", self.validate_auc)







#         for cur_epoch in range(1, total_epoch_num+1):
        
         
#             epoch_start_time = time.time()
#             self.cur_epoch = cur_epoch
#             self.train_epoch()
#             self.evaluate_epoch()
#             self.eval_train, self.eval_validate, self.training_auc, self.validate_auc  = self.train_top1.value(), self.eval_top1.value() , self.auc.value()[0], self.eval_auc.value()[0]
#             # printer
#             self.logger.log_printer.epoch_case_print(self.cur_epoch,
#                                                      self.train_top1.value(), self.eval_top1.value(),
#                                                      self.train_losses.avg,  self.eval_losses.avg,
#                                                      self.auc.value()[0], self.eval_auc.value()[0],
#                                                      time.time()-epoch_start_time)

#             # save best accuracy model
#             if self.eval_validate > best_acc:
#                 best_acc = self.eval_validate
#                 print('saving best acc model')
#                 self.model.save('False')
#             # save best auc model
#             if self.validate_auc > best_auc:
#                 best_auc = self.eval_validate
#                 print('saving best auc model')
#                 self.model.save('True')

#             #logger
#             self.logger.write_info_to_logger(variable_dict={'epoch':self.cur_epoch, 'lr':self.learning_rate,
#                                                             'train_acc':self.eval_train,
#                                                             'train_auc': self.training_auc,
#                                                             'validate_acc':self.eval_validate,
#                                                             'train_avg_loss':self.train_losses.avg,
#                                                             'validate_avg_loss':self.eval_losses.avg,
#                                                             'val_auc': self.validate_auc,
#                                                             'gpus_index': self.config['gpu_id'],
#                                                             #'save_name': os.path.join(self.config['save_path'],
#                                                              #                          self.config['save_name']),
#                                                             'net_name': self.config['model_net_name']})
#             self.logger.write()



#             wandb.log({
#                         "Epoch": cur_epoch,
#                         "Train Loss": self.train_losses.avg,
#                         "Train Acc":  self.train_top1.value(),
#                         "Train AUC":  self.auc.value()[0],
#                         "Valid Loss": self.eval_losses.avg,
#                         "Valid Acc":  self.eval_top1.value(),
#                         "Valid AUC":  self.eval_auc.value()[0]

# })
#             #tensorboard summary
#             if self.config['is_tensorboard']:
#                 self.logger.summarizer.data_summarize(self.cur_epoch, summarizer='train',
#                                                  summaries_dict={'train_acc': self.eval_train, 'train_avg_loss': self.train_losses.avg})
#                 self.logger.summarizer.data_summarize(self.cur_epoch, summarizer='validate',
#                                                  summaries_dict={'validate_acc': self.eval_validate,'validate_avg_loss': self.eval_losses.avg})
#                 if self.cur_epoch == total_epoch_num:
#                     self.logger.summarizer.graph_summary(self.model.net)
        
            # if cur_epoch == 5:
            #     break

    # def test(self):
        
    #     print("Testing started...")
        
            
    #     self.evaluate_test()

    #     self.eval_test, self.eval_auc = self.test_top1.avg, self.test_auc.avg

    #     print('acc: ', self.eval_test, 'auc:', self.eval_auc )
    #     wandb.log({
    #                     "acc": self.eval_test,
    #                     'auc:': self.eval_auc,
    #                  })

        # self.eval_test  = self.eval_test.avg, self.eval_top1.avg
        # print('\n')
        # for cur_epoch in range(1, total_epoch_num+1):
        #     print(cur_epoch)
         
        #     epoch_start_time = time.time()
        #     self.cur_epoch = cur_epoch
        #     self.train_epoch()
        #     self.evaluate_epoch()
                                # self.test_auc.add(probs.data, labels.data)
                                # prec1 = self.compute_accuracy(outputs, labels.data)
                                # self.eval_top1.update(prec1.item(), images.size(0))
        # self.test, self.eval_auc = self.train_top1.avg, self.eval_top1.avg
        # print(self.eval_test, self.eval_auc)
        #     # printer
        #     self.logger.log_printer.epoch_case_print(self.cur_epoch,
        #                                              self.train_top1.avg, self.eval_top1.avg,
        #                                              self.train_losses.avg,  self.eval_losses.avg,
        #                                              time.time()-epoch_start_time)


    # def train_epoch(self):
    #     """
    #     implement the logic of epoch:
    #     -loop ever the number of iteration in the config and call teh train step
    #     """
    #     raise NotImplementedError


    # def train_step(self):
    #     """
    #     implement the logic of the train step
    #     """
    #     raise NotImplementedError


    def evaluate_epoch(self):
        """
        implement the logic of epoch:
        -loop ever the number of iteration in the config and call teh train step
        """
        raise NotImplementedError


    def evaluate_step(self):
        """
        implement the logic of the train step
        """
        raise NotImplementedError


    def get_loss(self):
        """
        implement the logic of model loss
        """
        raise NotImplementedError


    # def create_optimization(self):
    #     """
    #     implement the logic of the optimization
    #     """
    #     raise NotImplementedError

    # def evaluate_test(self):
    #     """
    #     implement the logic of the test step
    #     """
    #     raise NotImplementedError


    # def evaluate_test_step(self):
    #     """
    #     implement the logic of the test step
    #     """
    #     raise NotImplementedError

        