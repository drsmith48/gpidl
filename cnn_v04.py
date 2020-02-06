#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:35:32 2019

@author: drsmith
"""

import sys, os, psutil, pickle, time, random, pathlib, contextlib
import datetime as dt
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader

GPIDATA = pathlib.Path(os.environ['HOME'])/'tigress/gpi/data'

class Net01(nn.Module):
    def __init__(self,
                 conv_layers=[6,16],
                 conv_size=[5,5],
                 fc_layers=[120,84],
                 activation='relu',
                 activ_kwargs={}):
        super().__init__()
        self.activation = getattr(nn.functional, activation)
        self.activ_kwargs = activ_kwargs
        # prepare convolution operations
        conv_layers = np.array(conv_layers, dtype=np.int)
        conv_size = np.array(conv_size, dtype=np.int)
        assert(conv_layers.size==conv_size.size)
        self.nconv = conv_layers.size
        datasize = np.array([64,80])
        for i in range(self.nconv):
            if i==0:
                conv = nn.Conv2d(1, conv_layers[i], conv_size[i])
            else:
                conv = nn.Conv2d(conv_layers[i-1], conv_layers[i], conv_size[i])
            attrname = 'conv{}'.format(i+1)
            setattr(self, attrname, conv)
            # shrink and decimate by 2 for each conv. layer
            datasize = (datasize-conv_size[i]+1) // 2
        # prepare fully-connected layer operations
        # datasize after conv/maxpool steps
        self.ndata = conv_layers[-1] * datasize.prod()
        fc_layers = np.array(fc_layers, dtype=np.int)
        for i in range(fc_layers.size):
            if i==0:
                fc = nn.Linear(self.ndata, fc_layers[i])
            else:
                fc = nn.Linear(fc_layers[i-1], fc_layers[i])
            attrname = 'fc{}'.format(i+1)
            setattr(self, attrname, fc)
        setattr(self, 'fc{}'.format(fc_layers.size+1), nn.Linear(fc_layers[-1], 4))
        self.nfc = fc_layers.size+1
        self.double()

    def forward(self, x):
        for i in range(self.nconv):
            conv_layer = getattr(self, 'conv{}'.format(i+1))
            x = conv_layer(x)
            x = self.activation(x, **self.activ_kwargs)
            x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, self.ndata)
        for i in range(self.nfc):
            fc_layer = getattr(self, 'fc{}'.format(i+1))
            x = fc_layer(x)
            x = self.activation(x, **self.activ_kwargs)
        return x


def trainnet(data=0,
             device=None,
             model_kwargs={},
             ftest=0.04,
             fvalidate=0.16,
             batch_size=4,
             superbatch=200,
             epochs=4,
             optim_name='SGD',
             optim_kwargs={'lr':1e-2},
             loss_name='CrossEntropyLoss',
             loss_kwargs={},
             scaling=True,
             repeat=1):

    # set data file and import data
    if data==1:
        filename = 'frame_category_1.pickle'
    elif data==2:
        filename = 'frame_category_2.pickle'
    else:
        filename = 'frame_category_small.pickle'
    filename = GPIDATA / filename
    print('Data file: {}'.format(filename.as_posix()))
    with filename.open('rb') as f:
        obj = pickle.load(f)
    frames = obj['frames'][:,np.newaxis,...].astype(np.float)
    true_labels = obj['frameinfo']['category'].to_numpy()
    assert(frames.shape[0]==true_labels.shape[0])
    
    # normalize frames
    if scaling:
        print('Frame-wise scaling to max(frame)=1')
        frame_max = frames.max(axis=(1,2,3))
        frames = frames / frame_max.reshape(-1,1,1,1)
    else:
        print('No scaling applied to frames')
    
    # frame stats
    nframes = frames.shape[0]
    print('Total frames: {}'.format(nframes))
    def fn_over_frames(fn_name, frames):
        fn = getattr(torch, fn_name)
        tmp1 = fn(frames,-1)
        if issubclass(type(tmp1),tuple):
            tmp1 = tmp1[0]
        tmp2 = fn(tmp1,-1)
        if issubclass(type(tmp2),tuple):
            tmp2 = tmp2[0]
        return tmp2
    allframes = torch.squeeze(torch.from_numpy(frames))
    for fn_name in ['min','max','mean','std','sum']:
        vals = fn_over_frames(fn_name, allframes)
        print('min/mean/max of frame-wise {}: {:.2f} {:.2f} {:.2f}'.
            format(fn_name, vals.min().item(), vals.mean().item(), vals.max().item()))
    del(allframes)
    ntest = np.int(nframes * ftest)
    nvalidate = np.int(nframes * fvalidate)
    ntrain = nframes - ntest - nvalidate
    print('Train/test/validate frames: {}/{}/{}'.format(ntrain,ntest,nvalidate))

    # device (CPU or GPU)
    if device:
        device = torch.device(device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device: ', device)
    
    # initiate model and send to device
    net = Net01(**model_kwargs)
    print('Model kwargs:', model_kwargs)
    net.to(device)
    print(list(net.modules())[0])
    # model parameters
    nparams = np.sum(np.array([p.numel() for p in net.parameters() if p.requires_grad]))
    print('Total model parameters: ', nparams)
    # define loss function
    lossClass = getattr(nn, loss_name)
    print('Loss function:', loss_name)
    print('Loss kwargs:', loss_kwargs)
    loss_function = lossClass(**loss_kwargs)
    # define optimizer
    optimizerClass = getattr(torch.optim, optim_name)
    print('Optimizer:', optim_name)
    print('Optimizer kwargs:', optim_kwargs)
    optimizer = optimizerClass(net.parameters(), **optim_kwargs)

    # loop over repeated trainings
    accuracy = np.empty(repeat)
    final_epoch = np.empty(repeat, dtype=np.int)
    print('Training runs:', repeat)
    for i in range(repeat):
    
        print('Training {}/{} with {} epochs and batch size {}'.
            format(i,repeat, epochs, batch_size))

        # shuffle frames, partition, and assemble datasets on device
        indices = np.arange(nframes)
        np.random.shuffle(indices)
        itrain = indices[:ntrain]
        tmp = np.delete(indices, np.arange(ntrain))
        itest = tmp[:ntest]
        tmp = np.delete(tmp, np.arange(ntest))
        assert(tmp.size==nvalidate)
        ivalidate = tmp[:]
        train_set = TensorDataset(torch.from_numpy(frames[itrain,...]),
                                  torch.from_numpy(true_labels[itrain]))
        train_loader = DataLoader(train_set, batch_size=batch_size)
        test_frames = torch.from_numpy(frames[itest,...]).to(device)
        test_labels = torch.from_numpy(true_labels[itest]).to(device)
        validate_frames = torch.from_numpy(frames[ivalidate,...]).to(device)
        validate_labels = torch.from_numpy(true_labels[ivalidate]).to(device)
        
        # reset (randomize) model parameters
        for mod in net.modules():
            if hasattr(mod, 'reset_parameters'):
                print('  Resetting params for module:', mod)
                mod.reset_parameters()

        # begin training loop over epochs
        test_losses = np.empty([0])
        epoch_break = False
        for epoch in range(epochs):
            for ibatch, data in enumerate(train_loader):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                if ibatch%superbatch == 0:
                    with torch.no_grad():
                        test_outputs = net(test_frames)
                        test_loss = loss_function(test_outputs, test_labels)
                        print('  Epoch {}/{}  batch {:06d}  test loss {:.6e}'.
                              format(epoch,epochs,ibatch,test_loss.item()))
                        test_losses = np.append(test_losses, test_loss.item())
                        if test_losses.size>35:
                            curr_loss = test_losses[-5:].mean()
                            past_loss = test_losses[-30:-25].mean()
                            if curr_loss >= 0.99*past_loss:
                                print('  LOSS NOT DECREASING, BREAKING')
                                epoch_break = True
                                break
            if epoch_break:
                break
        final_epoch[i] = epoch+1
        print('  End training in epoch', final_epoch[i])
        
        with torch.no_grad():
            validate_outputs = net(validate_frames)
            validate_loss = loss_function(validate_outputs, validate_labels)
            print('  Validation loss {:.3e}'.format(validate_loss.item()))
            _, predicted = torch.max(validate_outputs.data, 1)
            ncorrect = np.count_nonzero(predicted.cpu()==validate_labels.cpu())
            ntotal = validate_labels.size()[0]
            accuracy[i] = ncorrect/ntotal*100
            print('  Accuracy: {}/{} ({:.2f}%)'.format(ncorrect, ntotal, accuracy[i]))
            
        # end loop over repeat trainings
        
    print('End training loop')
    print('Accuracies:', accuracy)
            
    result = {'accuracy':accuracy,
              'final_epoch':final_epoch}

    return result


if __name__=="__main__":
    trainnet()
