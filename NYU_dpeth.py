# -*- coding: utf-8 -*-
# @Time    : 2018/10/21 1:01
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com

import torch
from torch.utils.data import DataLoader, dataset
from dataloaders import dataloader, nyu_dataloader
import os
import numpy as np


def NYUDepth_loader(data_path, batch_size = 32, isTrain = True):
    if isTrain:
        traindir = os.path.join(data_path, 'train')
        print(traindir)

        if os.path.exists(traindir):
            print('训练集目录存在')
        trainset = nyu_dataloader.NYUDataset(traindir, type='train')
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size = batch_size, shuffle=True)  # @wx 多线程读取失败
        return train_loader
    else:
        valdir = os.path.join(data_path, 'val')
        print(valdir)

        if os.path.exists(valdir):
            print('测试集目录存在')
        valset = nyu_dataloader.NYUDataset(valdir, type='val')
        val_loader = torch.utils.data.DataLoader(
            valset, batch_size = 1, shuffle=False # shuffle 测试时是否设置成False batch_size 恒定为1
        )
        return val_loader







