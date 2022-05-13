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
import numpy as np

class ExampleTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader,  config, logger, time, suffix, wandb_mode):
        super(ExampleTrainer, self).__init__(model, train_loader, val_loader, config, logger, time, suffix, wandb_mode)

        self.create_optimization()
        self.eval_auc = utils.AUCMeter()
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(self.config['patience']* 0.8), gamma=0.5)
        # self.learning_rate = float(*self.scheduler.get_last_lr())

        opt_level = 'O2'
        print(self.model)
        # asd?

        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=opt_level)


        # if args.exp_name is not None:
        #     self.set_exp_name(args.exp_name)
        # config = wandb.config

    def train_epoch(self):
        """
        training in a epoch
        :return: 
        """
        #print('in train_epoch')
        # Learning rate adjustment
        self.learning_rate = self.adjust_learning_rate(self.optimizer, self.cur_epoch)
        self.train_losses = utils.AverageMeter()
        self.train_top1 = utils.BalancedAccuracyMeter()
        self.auc = utils.AUCMeter()
        #self.auc.reset()
        # Set the model to be in training mode (for dropout and batchnorm)



        
        self.model.net.train()
        for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
            # print('batch_x ',batch_x.shape)
            # print('batch_y ',[batch_y[0].shape, batch_y[1].shape])
            # print(batch_y)
            if torch.cuda.is_available():
                batch_x = batch_x.cuda(non_blocking=self.config['async_loading'])
                batch_y1 = batch_y[0].cuda(non_blocking=self.config['async_loading'])
                batch_y2 = batch_y[1].cuda(non_blocking=self.config['async_loading'])
            batch_x_var, batch_y1_var, batch_y2_var = Variable(batch_x), Variable(batch_y1), Variable(batch_y2)
            # print('batch_x_var ',batch_x_var.shape)
            # print('batch_y1_var ',batch_y1_var.shape)
            # print('batch_y2_var ',batch_y2_var.shape)
            # asd
            self.train_step(batch_x_var, (batch_y1_var, batch_y2_var))
            # break
            
            # printer
            self.logger.log_printer.iter_case_print(self.cur_epoch, self.eval_train, self.eval_validate,
                                                    len(self.train_loader), batch_idx+1, self.train_losses.avg, self.learning_rate, self.auc.value()[0], self.eval_auc.value()[0])
        
            # if batch_idx == 2:
            #     break
            #tensorboard summary
            if self.config['is_tensorboard']:
                self.logger.summarizer.data_summarize(batch_idx, summarizer="train", summaries_dict={"lr":self.learning_rate, 'train_loss':self.train_losses.avg, 'auc': self.auc.value()[0]
                                               , 'eval_auc': self.eval_auc.value()[0]})

            
        # self.wandb_run.log({"train_loss": loss, 'train_accuracy': self.train_top1.value(), "train_auc": self.auc.value()[0], co})

        time.sleep(1)


    def train_step(self, images, labels):
        """
        training in a step
        :param images: 
        :param labels: 
        :return: 
        """
        labels1 = labels[0]
        labels2 = labels[1]
        # wandb.init(project="stoic_m", entity='mayug')
        # Initialize WandB 
   
        # Forward pass
        # print('images ', [images.min(), images.max(),'image shape', images.shape])
        # asd
        # torch.save(images.detach().cpu(), './temp/train_inp.pt')

        infer = self.model.forward(images)
        # print(infer)
        # print('predictions', infer.shape)
        # print('predictions', infer)
        # print('labels', labels)
        # Loss function

        # print('labels ', labels)
        # print('infer ', infer.shape)
        
        
        losses = self.get_loss(infer[:,0],labels1.float()) + 0.75 * self.get_loss(infer[:, 1],labels2.float())


        loss = losses.item()#.data[0]
        # print('loss ', loss)
        # asd
        
        # measure accuracy and record loss
        thresh=0.5
        probs = F.sigmoid(infer[:,0])
        # print('probs ', probs)
        
        outputs = (probs.data>thresh).long()
        print('outputs ', outputs)
        print('labels ', labels1)
        
        #prec1 = self.compute_accuracy(outputs, labels.data)
        # prec1 = self.compute_accuracy(outputs, labels.data, balanced = 'True')
        #print("accuracy of epoch", prec1)
        self.auc.add(probs.data, labels1.data)
        #print("auc of epoch", self.auc.value()[0])
        #print('images.size(0)', images.size(0), 'losss', loss)
        self.train_losses.update(loss, images.size(0))

        self.train_top1.update(outputs, labels1.data)

        #self.auc.update(self.auc.value()[0], images.size(0))


        

        #wandb
        
       
        # self.train_top5.update(prec5[0], images.size(0))
        # Optimization step
        if torch.cuda.device_count() > 1 and torch.cuda.is_available():
            self.optimizer.module.zero_grad()
        else:
            self.optimizer.zero_grad()
        with amp.scale_loss(losses, self.optimizer) as scaled_loss:
            scaled_loss.backward()

        # losses.backward()
        if torch.cuda.device_count() > 1 and torch.cuda.is_available():
            self.optimizer.module.step()
        else:
            self.optimizer.step()
    

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


    def create_optimization(self):
        """
        optimizer
        :return: 
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.config['learning_rate'], weight_decay=0.0005) #lr:1e-4
        # self.optimizer = torch.optim.SGD(self.model.net.parameters(), self.config['learning_rate'], momentum=0.9, weight_decay=0.0, nesterov=False)
        if torch.cuda.device_count() > 1:
            print('optimizer device_count: ',torch.cuda.device_count())
            self.optimizer = nn.DataParallel(self.optimizer,device_ids=range(torch.cuda.device_count()))
        """
        # optimizing parameters seperately
        ignored_params = list(map(id, self.model.net.fc.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params,
                            self.model.net.parameters())
        self.optimizer = torch.optim.Adam([
            {'params': base_params},
            {'params': self.model.net.fc.parameters(), 'lr': 1e-3}
            ], lr=1e-2, betas=(0.9, 0.99), eps=1e-08, weight_decay=0, amsgrad=False)"""


    def adjust_learning_rate(self, optimizer, epoch):
        """
        decay learning rate
        :param optimizer: 
        :param epoch: the first epoch is 1
        :return: 
        """
        # """Decay Learning rate at 1/2 and 3/4 of the num_epochs"""
        # lr = lr_init
        # if epoch >= num_epochs * 0.75:
        #     lr *= decay_rate ** 2
        # elif epoch >= num_epochs * 0.5:
        #     lr *= decay_rate
        learning_rate = self.config['learning_rate'] * (self.config['learning_rate_decay'] ** ((epoch - 1) // self.config['learning_rate_decay_epoch']))
        #print(learning_rate)
        if torch.cuda.device_count() > 1 and torch.cuda.is_available():
            for param_group in optimizer.module.param_groups:
                param_group['lr'] = learning_rate
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        return learning_rate


    # def compute_accuracy(self, output, target, topk=(1,), balanced = 'True'):
    #     """
    #     compute top-n accuracy
    #     :param output: 
    #     :param target: 
    #     :param topk: 
    #     :return: 
    #     """
        
    #     if len(topk)==1 and balanced == 'True':
    #         output = output.cpu().numpy()
    #         target = target.cpu().numpy()
    #         return  balanced_accuracy_score(output, target)

    #     if len(topk)==1 and balanced == 'False':
    #         return  torch.sum(output == target)/len(output)
    #     maxk = max(topk)
       
    #     batch_size = target.size(0)
    #     _, idx = output.topk(maxk, 1, True, True)
    #     idx = idx.t()
    #     correct = idx.eq(target.view(1, -1).expand_as(idx))
    #     acc_arr = []
    #     for k in topk:
    #         correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    #         acc_arr.append(correct_k.mul_(1.0 / batch_size))
    #     return acc_arr





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
        for batch_idx, (batch_x, batch_y) in enumerate(self.val_loader):
            batch_y = batch_y[0]
            if torch.cuda.is_available():
                #batch_x, batch_y = batch_x.cuda(async=self.config['async_loading']), batch_y.cuda(async=self.config['async_loading'])
                batch_x, batch_y = batch_x.cuda(non_blocking=self.config['async_loading']), batch_y.cuda(non_blocking=self.config['async_loading'])
            batch_x_var, batch_y_var = Variable(batch_x), Variable(batch_y)
            self.evaluate_step(batch_x_var, batch_y_var)
            # if batch_idx == 2:
            #     break
            # print('len', len(self.val_loader))
            # print('batch', batch_idx)
            utils.view_bar(batch_idx+1, len(self.val_loader))
        # self.wandb_run.log({"eval_loss": loss, 'eval_accuracy': eval_top1.value(), "eval_auc": self.eval_auc.value()[0]})


            
    def evaluate_step(self, images, labels):
        """
        evaluating in a step
        :param images: 
        :param labels: 
        :return: 
        """
        
        
        with torch.no_grad():
            # torch.save(images.detach().cpu(), './temp/val_inp.pt')
            # asd
            infer = self.model.forward(images)
            # label to one_hot
            # ids = labels.long().view(-1, 1)
            # one_hot_labels = torch.zeros(32, 2).scatter_(dim=1, index=ids, value=1.)
            # Loss function
            losses = self.get_loss(infer[:,0], labels.float())
            loss = losses.item()#losses.data[0]
        # print('infer: ', infer)
        # print('infer_data: ', infer.shape)
        # print(batch_idx)
        
        # measure accuracy and record loss
        probs = F.sigmoid(infer[:,0])
        # print('probs ', probs)
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


    def evaluate_epoch_leaderboard_auc(self):
        """
        evaluating in a epoch
        :return: 
        """
       
        self.eval_losses = utils.AverageMeter()
        self.eval_top1 = utils.BalancedAccuracyMeter()
        self.eval_auc = utils.AUCMeter()


        
        #self.auc.reset()
        # Set the model to be in testing mode (for dropout and batchnorm)
        self.model.eval()
        for batch_idx, (batch_x, batch_y) in enumerate(self.val_loader):
            if torch.cuda.is_available():
                #batch_x, batch_y = batch_x.cuda(async=self.config['async_loading']), batch_y.cuda(async=self.config['async_loading'])
                batch_x = batch_x.cuda(non_blocking=self.config['async_loading'])
                batch_y1 = batch_y[0].cuda(non_blocking=self.config['async_loading'])
                batch_y2 = batch_y[1].cuda(non_blocking=self.config['async_loading'])
            batch_x_var = Variable(batch_x)
            batch_y1_var, batch_y2_var = Variable(batch_y1), Variable(batch_y2)
            self.evaluate_step_leaderboard_auc(batch_x_var, (batch_y1_var, batch_y2_var))
            # if batch_idx == 2:
            #     break
            # print('len', len(self.val_loader))
            # print('batch', batch_idx)
            utils.view_bar(batch_idx+1, len(self.val_loader))



    def evaluate_step_leaderboard_auc(self, images, labels):
        """
        evaluating in a step
        :param images: 
        :param labels: 
        :return: 
        """
        # torch.save(images.detach().cpu(), '/temp/val_ct.pt')
        # asd
        labels1 = labels[0]
        labels2 = labels[1]

        print('images ', images.shape)
        b = images.shape[0]
        with torch.no_grad():
            infer = self.model(images)
        print('infer ', infer.shape)
        # measure accuracy and record loss
        probs = F.sigmoid(infer[:,0])
        print('probs ', probs)
        thresh=0.5
        outputs = (probs.data>thresh).long()
        
        print('inside eval')
        print('outputs ', outputs)
        # print(batch_idx)
        print('labels 0', labels)
        print('probs ', probs)
        # prec1 = self.compute_accuracy(outputs, labels.data, balanced = 'True')
        
        self.eval_top1.update(outputs, labels1.data)


        self.eval_auc.add(probs.data, labels1.data, positive_targets=labels2.data)

        # for name, prob in zip(names, F.sigmoid(infer).data):
            #     self.save_dict[name] = prob
  
    def test(self, positive_auc=False):

        if positive_auc:
            self.evaluate_epoch_leaderboard_auc()
        else:
            self.evaluate_epoch()       
        self.eval_top1, self.validate_auc = self.eval_top1.value(), self.eval_auc.value()[0]
        print('accuracy:', self.eval_top1, "auc:", self.validate_auc)
      

    # def evaluate_test(self):
    #     """
    #     testing
    #     :return: 
    #     """
    
    #     self.test_top1 = utils.AverageMeter()
    #     self.test_auc = utils.AverageMeter()
    #     self.auc = utils.AUCMeter()
    #     self.auc.reset()
    #     # Set the model to be in testing mode (for dropout and batchnorm)
    #     self.model.net.eval()
    #     for batch_idx, (batch_x, batch_y) in enumerate(self.test_loader):
    #         if torch.cuda.is_available():
    #             #batch_x, batch_y = batch_x.cuda(async=self.config['async_loading']), batch_y.cuda(async=self.config['async_loading'])
    #             batch_x, batch_y = batch_x.cuda(non_blocking=self.config['async_loading']), batch_y.cuda(non_blocking=self.config['async_loading'])
    #         batch_x_var, batch_y_var = Variable(batch_x), Variable(batch_y)

    #         self.evaluate_test_step(batch_x_var, batch_y_var)
        
    #         #if batch_idx == 1:
    #         #    break
    #         utils.view_bar(batch_idx+1, len(self.test_loader))
            
        


    # def evaluate_test_step(self, images, labels):
    #     """
    #     evaluating in a step
    #     :param images: 
    #     :param labels: 
    #     :return: 
    #     """

    #     with torch.no_grad():

            
    #         infer = self.model.forward(images)
            
    #     probs = F.sigmoid(infer)
    #     # print('probs ', probs)
    #     thresh=0.5
    #     outputs = (probs.data>thresh).long()

    #     self.auc.add(probs.data, labels.data)
    #     prec1 = self.compute_accuracy(outputs, labels.data, balanced = 'True')
    #     self.test_top1.update(prec1.item(), images.size(0))
    
    #     self.test_auc.update(self.auc.value()[0], images.size(0))
            
        



