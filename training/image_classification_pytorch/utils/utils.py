# coding=utf-8
from __future__ import print_function
import numpy as np
import time
import sys
import os
import torch
import numbers
from sklearn.metrics import balanced_accuracy_score

class Meter(object):
    '''Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    '''

    def reset(self):
        '''Resets the meter to default settings.'''
        pass

    def add(self, value):
        '''Log a new value to the meter
        Args:
            value: Next restult to include.
        '''
        pass

    def value(self):
        '''Get the value of the meter in the current state.'''
        pass

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class BalancedAccuracyMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):

        self.outputs= [] 
        self.targets = []


    def update(self, output, target):
        self.outputs.extend(list(output.cpu().numpy()))
        self.targets.extend(list(target.cpu().numpy()))

        # print('out ', self.outputs)
        # print('target ', self.targets)

    def value(self):
        outputs = np.array(self.outputs)
        targets = np.array(self.targets)
        return balanced_accuracy_score(outputs, targets)
        




# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res


def view_bar(num, total):
    """
    
    :param num: 
    :param total: 
    :return: 
    """
    rate = float(num + 1) / total
    rate_num = int(rate * 100)

   
    if num != total:
        r = '\r[%s%s]%d%%' % ("=" * rate_num, " " * (100 - rate_num), rate_num,)
    else:
        r = '\r[%s%s]%d%%' % ("=" * 100, " " * 0, 100,)
    sys.stdout.write(r)
    sys.stdout.flush()


class AUCMeter(Meter):

    def __init__(self):
            super(AUCMeter, self).__init__()
            self.reset()

    def reset(self):
            self.scores = torch.DoubleTensor(torch.DoubleStorage()).numpy()
            self.targets = torch.LongTensor(torch.LongStorage()).numpy()



    def __init__(self):
            super(AUCMeter, self).__init__()
            self.reset()

    def reset(self):
            self.scores = torch.DoubleTensor(torch.DoubleStorage()).numpy()
            self.targets = torch.LongTensor(torch.LongStorage()).numpy()

    # def add(self, output, target):
    #         if torch.is_tensor(output):
    #             output = output.cpu().squeeze().numpy()
    #         if torch.is_tensor(target):
    #             target = target.cpu().squeeze().numpy()
    #         elif isinstance(target, numbers.Number):
    #             target = np.asarray([target])
    #         assert np.ndim(output) == 1, \
    #             'wrong output size (1D expected)'
    #         assert np.ndim(target) == 1, \
    #             'wrong target size (1D expected)'
    #         assert output.shape[0] == target.shape[0], \
    #             'number of outputs and targets does not match'
    #         assert np.all(np.add(np.equal(target, 1), np.equal(target, 0))), \
    #             'targets should be binary (0, 1)'

    #         self.scores = np.append(self.scores, output)
    #         self.targets = np.append(self.targets, target)
    def add(self, output, target, positive_targets=None):
            if torch.is_tensor(output):
                output = output.cpu().numpy()
            if torch.is_tensor(target):
                target = target.cpu().numpy()
            elif isinstance(target, numbers.Number):
                target = np.asarray([target])

            # print([np.ndim(output), np.ndim(target)])
            # print([output, target])
            assert np.ndim(output) == 1, \
                'wrong output size (1D expected)'
            assert np.ndim(target) == 1, \
                'wrong target size (1D expected)'
            assert output.shape[0] == target.shape[0], \
                'number of outputs and targets does not match'
            assert np.all(np.add(np.equal(target, 1), np.equal(target, 0))), \
                'targets should be binary (0, 1)'

            if positive_targets is not None:
                positive_targets = positive_targets.cpu().numpy()
                print('before ', [output.shape, target.shape])
                output = output[positive_targets==1]
                target = target[positive_targets==1]
                print('after ', [output.shape, target.shape])

            self.scores = np.append(self.scores, output)
            self.targets = np.append(self.targets, target)

    def value(self):
            # case when number of elements added are 0
            if self.scores.shape[0] == 0:
                return (0.5, 0.0, 0.0)

            # sorting the arrays
            scores, sortind = torch.sort(torch.from_numpy(
                self.scores), dim=0, descending=True)
            scores = scores.numpy()
            sortind = sortind.numpy()

            # creating the roc curve
            tpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)
            fpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)

            for i in range(1, scores.size + 1):
                if self.targets[sortind[i - 1]] == 1:
                    tpr[i] = tpr[i - 1] + 1
                    fpr[i] = fpr[i - 1]
                else:
                    tpr[i] = tpr[i - 1]
                    fpr[i] = fpr[i - 1] + 1

            tpr /= (self.targets.sum() * 1.0)
            fpr /= ((self.targets - 1.0).sum() * -1.0)

            # calculating area under curve using trapezoidal rule
            n = tpr.shape[0]
            h = fpr[1:n] - fpr[0:n - 1]
            sum_h = np.zeros(fpr.shape)
            sum_h[0:n - 1] = h
            sum_h[1:n] += h
            area = (sum_h * tpr).sum() / 2.0

            return (area, tpr, fpr)