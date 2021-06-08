#!/usr/bin/env python3
# from __future__ import print_function, absolute_import
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Functions for manipulating networks."""

import itertools
from pycls.core.config import cfg
import pycls.core.distributed as dist
import time
import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy.stats import entropy
from math import log, e
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D  # .multivariate_normal import MultivariateNormal
import torch


class CELossAndHLossAndFLoss(nn.Module):
    def __init__(self):
        super(CELossAndHLossAndFLoss, self).__init__()
        # torch.nn.loss.py
        self.CELoss = nn.CrossEntropyLoss()
        self.L1Loss = nn.L1Loss()
        self.MSELoss = nn.MSELoss()
        self.SmoothL1Loss = nn.SmoothL1Loss()
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
        # My own loss function
        self.HLoss = HLoss()
        self.FLoss = FLoss()
        self.FLoss2 = FLoss2()
        self.HLossNoSoftMax = HLossNoSoftMax()
        self.FLossNoSoftMax = FLossNoSoftMax()
        self.MultivariateNormalLoss = MultivariateNormalLoss()
        pass

    def ComputeCELossMultivariateNormalLoss(self, **kwargs):
        loss = self.CELoss(kwargs['outputs'], kwargs['targets']) \
               + kwargs['sigma'] * self.MultivariateNormalLoss(kwargs['top_c'], kwargs['outputs'])
        return loss

        pass

    def ComputeCELossFLoss2(self, **kwargs):
        loss = self.CELoss(kwargs['outputs'], kwargs['targets']) \
               + kwargs['theta'] * self.FLoss2(kwargs['top_c'], kwargs['outputs'], kwargs['targets'])
        return loss
        pass

    def ComputeCELossMultivariateNormalLossFLoss2(self, **kwargs):
        loss = self.CELoss(kwargs['outputs'], kwargs['targets']) \
               + kwargs['theta'] * self.FLoss2(kwargs['top_c'], kwargs['outputs'], kwargs['targets'])\
               + kwargs['sigma'] * self.MultivariateNormalLoss(kwargs['top_c'], kwargs['outputs'])
        return loss
        pass

    def ComputeL1Loss(self, **kwargs):
        loss = self.L1Loss(kwargs['outputs'], kwargs['targets'])
        return loss
        pass

    def ComputeSmoothL1Loss(self, **kwargs):
        loss = self.SmoothL1Loss(kwargs['outputs'], kwargs['targets'])
        return loss
        pass

    def ComputeBCEWithLogitsLoss(self, **kwargs):
        loss = self.BCEWithLogitsLoss(kwargs['outputs'], kwargs['targets'])
        return loss
        pass

    def ComputeMSELossAndHLoss(self, **kwargs):
        loss = self.MSELoss(kwargs['outputs'], kwargs['targets']) + kwargs['lamda'] * self.HLoss(kwargs['outputs'])
        return loss
        pass

    def ComputeMSELoss(self, **kwargs):
        loss = self.MSELoss(kwargs['outputs'], kwargs['targets'])
        return loss
        pass

    def ComputeMSELossAndFLoss(self, **kwargs):
        loss = self.MSELoss(kwargs['outputs'], kwargs['targets']) \
               + kwargs['theta'] * self.FLoss(kwargs['top_c'], kwargs['outputs'])
        return loss
        pass

    def ComputeMSELossAndHLossAndFLoss(self, **kwargs):
        loss = self.MSELoss(kwargs['outputs'], kwargs['targets']) + kwargs['lamda'] * self.HLoss(kwargs['outputs'])\
               + kwargs['theta'] * self.FLoss(kwargs['top_c'], kwargs['outputs'])
        return loss
        pass

    def ComputeCELossAndHLoss(self, **kwargs):
        loss = self.CELoss(kwargs['outputs'], kwargs['targets']) + kwargs['lamda'] * self.HLoss(kwargs['outputs'])
        return loss
        pass

    def ComputeCELoss(self, **kwargs):
        loss = self.CELoss(kwargs['outputs'], kwargs['targets'])
        return loss
        pass

    def ComputeCELossAndFLoss(self, **kwargs):
        loss = self.CELoss(kwargs['outputs'], kwargs['targets']) \
               + kwargs['theta'] * self.FLoss(kwargs['top_c'], kwargs['outputs'])
        return loss
        pass

    def ComputeHLossAndFLoss(self, **kwargs):
        loss = kwargs['lamda'] * self.HLoss(kwargs['outputs'])\
               + kwargs['theta'] * self.FLoss(kwargs['top_c'], kwargs['outputs'])
        return loss
        pass

    def ComputeHLossAndFLossNoSoftMax(self, **kwargs):
        loss = kwargs['lamda'] * self.HLoss(kwargs['outputs'])\
               + kwargs['theta'] * self.FLossNoSoftMax(kwargs['top_c'], kwargs['outputs'])
        return loss
        pass

    def ComputeCELossAndHLossAndFLoss(self, **kwargs):
        loss = self.CELoss(kwargs['outputs'], kwargs['targets']) + kwargs['lamda'] * self.HLoss(kwargs['outputs'])\
               + kwargs['theta'] * self.FLoss(kwargs['top_c'], kwargs['outputs'])
        return loss
        pass

    def ComputeCELossAndHLossNoSoftMax(self, **kwargs):
        loss = self.CELoss(kwargs['outputs'], kwargs['targets']) + kwargs['lamda'] * self.HLossNoSoftMax(kwargs['outputs'])
        return loss
        pass

    def ComputeCELossAndFLossNoSoftMax(self, **kwargs):
        loss = self.CELoss(kwargs['outputs'], kwargs['targets']) \
               + kwargs['theta'] * self.FLossNoSoftMax(kwargs['top_c'], kwargs['outputs'])
        # print("FLoss_no_soft: ", '\n', kwargs['theta'] * self.FLossNoSoftMax(kwargs['top_c'], kwargs['outputs']))
        return loss
        pass

    def ComputeHLossNoSoftMaxAndFLossNoSoftMax(self, **kwargs):
        loss = kwargs['lamda'] * self.HLossNoSoftMax(kwargs['outputs'])\
               + kwargs['theta'] * self.FLossNoSoftMax(kwargs['top_c'], kwargs['outputs'])
        return loss
        pass

    def ComputeCELossAndHLossNoSoftMaxAndFLossNoSoftMax(self, **kwargs):
        loss = self.CELoss(kwargs['outputs'], kwargs['targets']) + kwargs['lamda'] * self.HLossNoSoftMax(kwargs['outputs'])\
               + kwargs['theta'] * self.FLossNoSoftMax(kwargs['top_c'], kwargs['outputs'])
        return loss
        pass

    def ComputeCELossAndHLossAndFLossNoSoftMax(self, **kwargs):
        loss = self.CELoss(kwargs['outputs'], kwargs['targets']) + kwargs['lamda'] * self.HLoss(kwargs['outputs'])\
               + kwargs['theta'] * self.FLossNoSoftMax(kwargs['top_c'], kwargs['outputs'])
        return loss
        pass
    pass


class FLossNoSoftMax(nn.Module):
    def __init__(self):
        super(FLossNoSoftMax, self).__init__()

    def forward(self, top_c, output):
        ambiguous_y = torch.zeros(output.shape).cuda()
        _, index_j = output.topk(top_c, 1, True, True)
        index_i = torch.ones(index_j.shape).cuda()
        ambiguous_y.scatter_(1, index_j, index_i)
        x1 = 1. - ambiguous_y
        x2 = 1. - output
        # loss = F.softmax(x1, dim=1) * F.log_softmax(x2, dim=1)
        loss = x1 * torch.log(x2)
        loss = torch.mean(loss, 1)
        loss = -1.0 * loss.sum()
        return loss


class HLossNoSoftMax(nn.Module):
    def __init__(self):
        super(HLossNoSoftMax, self).__init__()

    def forward(self, x):
        # loss = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        loss = x * torch.log(x)
        loss = torch.mean(loss, 1)
        loss = -1.0 * loss.sum()
        return loss


class FLoss(nn.Module):
    def __init__(self):
        super(FLoss, self).__init__()
        # ambiguous_y = torch.zeros(output.shape).cuda()

    def forward(self, top_c, output):
        ambiguous_y = torch.zeros(output.shape).cuda()
        _, index_j = output.topk(top_c, 1, True, True)
        index_i = torch.ones(index_j.shape).cuda()
        ambiguous_y.scatter_(1, index_j, index_i)
        x1 = 1. - ambiguous_y
        x2 = 1. - output
        loss = F.softmax(x1, dim=1) * F.log_softmax(x2, dim=1)
        # loss = x1 * F.log_softmax(x2, dim=1)
        loss = torch.mean(loss, 1)
        loss = -1.0 * loss.sum()
        return loss


class FLoss2(nn.Module):
    def __init__(self):
        super(FLoss2, self).__init__()
        pass

    def forward(self, top_c, output, target):
        BS = output.shape[0]  # 128
        target = target.view(BS, 1)
        # print(target.shape)
        output_c, index_c = output.topk(top_c, 1, True, True)
        target_c = torch.zeros(output.shape).cuda()
        index_i = torch.ones(target.shape).cuda()
        target_c.scatter_(1, target, index_i)

        index_i = torch.arange(0, BS, 1).view(BS, 1)
        index_i = index_i.expand(BS, top_c)  # (128, 3)

        # index_i = np.arange(BS, dtype=int).T
        # index_i = np.tile(index_i, (top_c, 1)).T
        target_cc = target_c[index_i, index_c]
        print('target_cc: \n', F.softmax(target_cc, dim=1))
        print('output_c: \n', F.softmax(output_c, dim=1))
        # cross entropy
        # todo: try MSE Loss
        # print(output_c.shape)
        # print(target_cc.shape)
        # loss_mse = nn.MSELoss()
        # print('mse loss : ', loss_mse(output_c, target_cc))
        # loss = F.softmax(target_cc, dim=1) * F.log_softmax(output_c, dim=1)
        loss = target_cc * F.log_softmax(output_c, dim=1)
        loss = torch.mean(loss, 1)
        loss = torch.where(loss != 0.0, loss, -2.3026 * torch.ones([BS]).cuda())
        print('loss: \n', loss)
        loss = -1.0 * loss.sum() / BS
        # print(loss)
        # exit()
        return loss


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        loss = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)  # e as base for log()
        loss = torch.mean(loss, 1)
        loss = -1.0 * loss.sum()
        return loss  # 0 <= loss <= log(n)/n*BS


"""
Using the entropy of output make the sigma of a Multi Variate Normal.
The entropy decrease，the MVN be more ease. 
The entropy crease, the MVN be more steep.

"""
class MultivariateNormalLoss(nn.Module):
    def __init__(self):
        super(MultivariateNormalLoss, self).__init__()
        pass

    def forward(self, top_c, output):
        hloss = F.softmax(output, dim=1) * F.log_softmax(output, dim=1)
        hloss = torch.mean(hloss, 1)

        hloss = (-1.0 * hloss.sum()) * output.shape[1] / (torch.log(torch.Tensor([output.shape[1]]).cuda()) * output.shape[0])

        output, _ = output.topk(top_c, 1, True, True)
        # print(output.shape)
        m = D.multivariate_normal.MultivariateNormal(torch.zeros(output.shape[1]).cuda(), hloss * torch.eye(output.shape[1]).cuda())
        # x = F.softmax(x, dim=1)
        # output = F.softmax(output, dim=1)
        loss = m.log_prob(output).exp().sum()
        return loss   # 0 <= loss <= log(n)/n*BS


def print_debug(_=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), parameter=None, is_debug=False):
    if is_debug:
        print('\n{:}: \'{:}\'\n'.format(_, parameter))
        pass
    pass


def entropy1(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)


def entropy2(labels, base=None):
    """ Computes entropy of label distribution. """

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)
        pass

    return ent


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    # print_debug('maxk', parameter=maxk, is_debug=True)
    # print_debug('batch_size', parameter=batch_size, is_debug=True)

    _, pred = output.topk(maxk, 1, True, True)
    # print('output.topk(maxk, 1, True, True)[0]\n', output.topk(maxk, 1, True, True)[0])
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def unwrap_model(model):
    """Remove the DistributedDataParallel wrapper if present."""
    wrapped = isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel)
    return model.module if wrapped else model


@torch.no_grad()
def compute_precise_bn_stats(model, loader):
    """Computes precise BN stats on training data."""
    # Compute the number of minibatches to use
    num_iter = int(cfg.BN.NUM_SAMPLES_PRECISE / loader.batch_size / cfg.NUM_GPUS)
    num_iter = min(num_iter, len(loader))
    # Retrieve the BN layers
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    # Initialize BN stats storage for computing mean(mean(batch)) and mean(var(batch))
    running_means = [torch.zeros_like(bn.running_mean) for bn in bns]
    running_vars = [torch.zeros_like(bn.running_var) for bn in bns]
    # Remember momentum values
    momentums = [bn.momentum for bn in bns]
    # Set momentum to 1.0 to compute BN stats that only reflect the current batch
    for bn in bns:
        bn.momentum = 1.0
    # Average the BN stats for each BN layer over the batches
    for inputs, _labels in itertools.islice(loader, num_iter):
        model(inputs.cuda())
        for i, bn in enumerate(bns):
            running_means[i] += bn.running_mean / num_iter
            running_vars[i] += bn.running_var / num_iter
    # Sync BN stats across GPUs (no reduction if 1 GPU used)
    running_means = dist.scaled_all_reduce(running_means)
    running_vars = dist.scaled_all_reduce(running_vars)
    # Set BN stats and restore original momentum values
    for i, bn in enumerate(bns):
        bn.running_mean = running_means[i]
        bn.running_var = running_vars[i]
        bn.momentum = momentums[i]


def complexity(model):
    """Compute model complexity (model can be model instance or model class)."""
    size = cfg.TRAIN.IM_SIZE
    cx = {"h": size, "w": size, "flops": 0, "params": 0, "acts": 0}
    cx = unwrap_model(model).complexity(cx)
    return {"flops": cx["flops"], "params": cx["params"], "acts": cx["acts"]}


def smooth_one_hot_labels(labels):
    """Convert each label to a one-hot vector."""
    n_classes, label_smooth = cfg.MODEL.NUM_CLASSES, cfg.TRAIN.LABEL_SMOOTHING
    err_str = "Invalid input to one_hot_vector()"
    assert labels.ndim == 1 and labels.max() < n_classes, err_str
    shape = (labels.shape[0], n_classes)
    neg_val = label_smooth / n_classes
    pos_val = 1.0 - label_smooth + neg_val
    labels_one_hot = torch.full(shape, neg_val, dtype=torch.long, device=labels.device)  # dtype=torch.float
    labels_one_hot.scatter_(1, labels.long().view(-1, 1), pos_val)
    return labels_one_hot


class SoftCrossEntropyLoss(torch.nn.Module):
    """SoftCrossEntropyLoss (useful for label smoothing and mixup).
    Identical to torch.nn.CrossEntropyLoss if used with one-hot labels."""

    def __init__(self):
        super(SoftCrossEntropyLoss, self).__init__()

    def forward(self, x, y):
        loss = -y * torch.nn.functional.log_softmax(x, -1)
        return torch.sum(loss) / x.shape[0]


def mixup(inputs, labels):
    """Apply mixup to minibatch (https://arxiv.org/abs/1710.09412)."""
    alpha = cfg.TRAIN.MIXUP_ALPHA
    assert labels.shape[1] == cfg.MODEL.NUM_CLASSES, "mixup labels must be one-hot"
    if alpha > 0:
        m = np.random.beta(alpha, alpha)
        permutation = torch.randperm(labels.shape[0])
        inputs = m * inputs + (1.0 - m) * inputs[permutation, :]
        labels = m * labels + (1.0 - m) * labels[permutation, :]
    return inputs, labels, labels.argmax(1)


def update_model_ema(model, model_ema, cur_epoch, cur_iter):
    """Update exponential moving average (ema) of model weights."""
    update_period = cfg.OPTIM.EMA_UPDATE_PERIOD
    if update_period == 0 or cur_iter % update_period != 0:
        return
    # Adjust alpha to be fairly independent of other parameters
    adjust = cfg.TRAIN.BATCH_SIZE / cfg.OPTIM.MAX_EPOCH * update_period
    alpha = min(1.0, cfg.OPTIM.EMA_ALPHA * adjust)
    # During warmup simply copy over weights instead of using ema
    alpha = 1.0 if cur_epoch < cfg.OPTIM.WARMUP_EPOCHS else alpha
    # Take ema of all parameters (not just named parameters)
    params = unwrap_model(model).state_dict()
    for name, param in unwrap_model(model_ema).state_dict().items():
        param.copy_(param * (1.0 - alpha) + params[name] * alpha)


if __name__ == "__main__":
    import time
    import os

    from torch.distributions.multivariate_normal import MultivariateNormal
    # torch.distributions.distribution.Distribution
    # m = MultivariateNormal(torch.zeros(10), 0.84*torch.eye(10))
    # m.log_prob(2)
    # print(m.log_prob(torch.randn(1, 10)).exp())
    # exit()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("####" * 25, '\n', 'Start\n')
    y_train = 0.1 * torch.ones(128, 10).cuda()
    # y_train = torch.randn(128, 10).cuda()
    print(F.softmax(y_train, dim=1))
    targets = torch.empty(128, dtype=torch.long).random_(10).cuda()  # torch.randn(128, dtype=torch.long).cuda()

    t1 = time.time()
    # print('outputs: \n', y_train.shape)
    # print('targets: \n', targets.shape, '\n', targets)

    # exit()
    # criterion = CELossAndHLossAndFLoss()

    criterion = FLoss2()
    criterion = MultivariateNormalLoss()
    celoss = nn.CrossEntropyLoss()
    print('CE loss: \n', celoss(y_train, targets)) # .view(-1)
    # criterion = FLoss2()
    # loss = criterion.HLoss(outputs=y_train, targets=targets, lamda=0.001, theta=0.1, top_c=3)
    # loss = criterion(3, y_train, targets)
    loss = criterion(3, y_train)
    print('loss: \n', loss)

    t2 = time.time()
    print("====" * 25, '\n', 'Take time:{:06f}'.format(t2 - t1) + 'S\nEnd')
    pass