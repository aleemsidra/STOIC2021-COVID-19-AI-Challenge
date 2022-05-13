# coding=utf-8
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from trainers.base_trainer import BaseTrainer
from utils import utils
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from apex import amp

class ExampleTrainer(BaseTrainer):
    def __init__(self, model, test_loader,  config, logger):
        super(ExampleTrainer, self).__init__(model, test_loader, config, logger)

        #self.create_optimization()
        self.eval_auc = utils.AUCMeter()
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(self.config['patience']* 0.8), gamma=0.5)
        # self.learning_rate = float(*self.scheduler.get_last_lr())

        opt_level = 'O2'
        print(self.model)
        # asd?

        # self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=opt_level)


    # def train_epoch(self):
    #     """
    #     training in a epoch
    #     :return: 
    #     """
    #     #print('in train_epoch')
    #     # Learning rate adjustment
    #     # self.learning_rate = self.adjust_learning_rate(self.optimizer, self.cur_epoch)
    #     self.train_losses = utils.AverageMeter()
    #     self.train_top1 = utils.BalancedAccuracyMeter()
    #     self.auc = utils.AUCMeter()
    #     #self.auc.reset()
    #     # Set the model to be in training mode (for dropout and batchnorm)



        
    #     self.model.net.train()
    #     for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
    #         if torch.cuda.is_available():
    #             batch_x, batch_y = batch_x.cuda(non_blocking=self.config['async_loading']), batch_y.cuda(non_blocking=self.config['async_loading'])
    #         batch_x_var, batch_y_var = Variable(batch_x), Variable(batch_y)
    #         self.train_step(batch_x_var, batch_y_var)
            
    #         # printer
    #         self.logger.log_printer.iter_case_print(self.cur_epoch, self.eval_train, self.eval_validate,
    #                                                 len(self.train_loader), batch_idx+1, self.train_losses.avg, self.learning_rate, self.auc.value()[0], self.eval_auc.value()[0])
        
    #         # if batch_idx == 2:
    #         #     break
    #         #tensorboard summary
    #         if self.config['is_tensorboard']:
    #             self.logger.summarizer.data_summarize(batch_idx, summarizer="train", summaries_dict={"lr":self.learning_rate, 'train_loss':self.train_losses.avg, 'auc': self.auc.value()[0]
    #                                            , 'eval_auc': self.eval_auc.value()[0]})
            
            

    #     time.sleep(1)


    # def train_step(self, images, labels):
    #     """
    #     training in a step
    #     :param images: 
    #     :param labels: 
    #     :return: 
    #     """
    #     #wandb.init(project="stoic", entity="sidra")
    #     # Initialize WandB 
   
    #     # Forward pass
    #     #print('images ', [images.min(), images.max(),'image shape', images.shape])

    #     infer = self.model.forward(images)
    #     # print(infer)
    #     # print('predictions', infer.shape)
    #     # print('predictions', infer)
    #     # print('labels', labels)
    #     # Loss function

    #     # print('labels ', labels)
    #     # print('infer ', infer)
        
    #     losses = self.get_loss(infer,labels.float())

    #     loss = losses.item()#.data[0]
    #     # print('loss ', loss)
        
    #     # measure accuracy and record loss
    #     thresh=0.5
    #     probs = F.sigmoid(infer)
    #     # print('probs ', probs)
        
    #     outputs = (probs.data>thresh).long()
    #     print('outputs ', outputs)
    #     print('labels ', labels)
        
    #     #prec1 = self.compute_accuracy(outputs, labels.data)
    #     # prec1 = self.compute_accuracy(outputs, labels.data, balanced = 'True')
    #     #print("accuracy of epoch", prec1)
    #     self.auc.add(probs.data, labels.data)
    #     #print("auc of epoch", self.auc.value()[0])
    #     #print('images.size(0)', images.size(0), 'losss', loss)
    #     self.train_losses.update(loss, images.size(0))

    #     self.train_top1.update(outputs, labels.data)

    #     #self.auc.update(self.auc.value()[0], images.size(0))


        

    #     #wandb
    #     #wandb.log({"train_loss": loss, 'train_accuracy': prec1, "train_auc": self.auc.value()[0]})
       
    #     # self.train_top5.update(prec5[0], images.size(0))
    #     # Optimization step
    #     if torch.cuda.device_count() > 1 and torch.cuda.is_available():
    #         self.optimizer.module.zero_grad()
    #     else:
    #         self.optimizer.zero_grad()
    #     with amp.scale_loss(losses, self.optimizer) as scaled_loss:
    #         scaled_loss.backward()

    #     # losses.backward()
    #     if torch.cuda.device_count() > 1 and torch.cuda.is_available():
    #         self.optimizer.module.step()
    #     else:
    #         self.optimizer.step()
    

    def get_loss(self, pred, label):
        """
        compute loss
        :param pred: 
        :param label: 
        :return: 
        """
        # criterion = nn.CrossEntropyLoss()  # nn.MSELoss()
        criterion = nn.BCEWithLogitsLoss()
        if torch.cuda.is_available():
            criterion.cuda()
        return criterion(pred, label)


    # def create_optimization(self):
    #     """
    #     optimizer
    #     :return: 
    #     """
    #     self.optimizer = torch.optim.Adam(self.model.net.parameters(),
    #                                       lr=self.config['learning_rate'], weight_decay=0) #lr:1e-4
    #     # self.optimizer = torch.optim.SGD(self.model.net.parameters(), self.config['learning_rate'], momentum=0.9, weight_decay=0.0, nesterov=False)
    #     if torch.cuda.device_count() > 1:
    #         print('optimizer device_count: ',torch.cuda.device_count())
    #         self.optimizer = nn.DataParallel(self.optimizer,device_ids=range(torch.cuda.device_count()))
    #     """
    #     # optimizing parameters seperately
    #     ignored_params = list(map(id, self.model.net.fc.parameters()))
    #     base_params = filter(lambda p: id(p) not in ignored_params,
    #                         self.model.net.parameters())
    #     self.optimizer = torch.optim.Adam([
    #         {'params': base_params},
    #         {'params': self.model.net.fc.parameters(), 'lr': 1e-3}
    #         ], lr=1e-2, betas=(0.9, 0.99), eps=1e-08, weight_decay=0, amsgrad=False)"""



    def evaluate_epoch(self):
        """
        evaluating in a epoch
        :return: 
        """
       
        self.eval_losses = utils.AverageMeter()
        self.eval_top1 = utils.BalancedAccuracyMeter()
        self.eval_auc = utils.AUCMeter()
        
        #self.auc.reset()
        # Set the model to be in testing mode (for dropout and batchnorm)
        self.model.net.eval()
        for batch_idx, (batch_x, batch_y) in enumerate(self.test_loader):
            if torch.cuda.is_available():
                #batch_x, batch_y = batch_x.cuda(async=self.config['async_loading']), batch_y.cuda(async=self.config['async_loading'])
                batch_x, batch_y = batch_x.cuda(non_blocking=self.config['async_loading']), batch_y.cuda(non_blocking=self.config['async_loading'])
            batch_x_var, batch_y_var = Variable(batch_x), Variable(batch_y)
            self.evaluate_step(batch_x_var, batch_y_var)
            # if batch_idx == 2:
            #     break
            # print('len', len(self.val_loader))
            # print('batch', batch_idx)
            utils.view_bar(batch_idx+1, len(self.test_loader))


            
    def evaluate_step(self, images, labels):
        """
        evaluating in a step
        :param images: 
        :param labels: 
        :return: 
        """
        
        with torch.no_grad():
            infer = self.model(images)
            losses = self.get_loss(infer, labels.float())
            loss = losses.item()#losses.data[0]
        # print('infer: ', infer)
        # print('infer_data: ', infer.shape)
        # print(batch_idx)
        
        # measure accuracy and record loss
        probs = F.sigmoid(infer)
        print('probs ', probs)
        thresh=0.5
        outputs = (probs.data>thresh).long()
        
        print('inside eval')
        print('outputs ', outputs)
        # print(batch_idx)
        print('labels 0', labels)
        # prec1 = self.compute_accuracy(outputs, labels.data, balanced = 'True')
        self.eval_top1.update(outputs, labels.data)
        self.eval_losses.update(loss, images.size(0)) # loss.data[0]
        self.eval_auc.add(probs.data, labels.data)

 
        



