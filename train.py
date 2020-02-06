#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle, pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import CNN

gpidir = pathlib.Path.home()/'gpi'

def train(data=0,
          device=None,
          model_kwargs={},
          ftest=0.04,
          fvalidate=0.16,
          batch_size=4,
          superbatch=200,
          epochs=2,
          optim_name='Adamax',
          optim_kwargs={'lr':1e-3},
          loss_name='CrossEntropyLoss',
          loss_kwargs={},
          scaling=True,
          repeat=3):

    # set data file and import data
    datadir = gpidir/'data'
    if data==1:
        filename = datadir/'frame_category_1.pickle'
    elif data==2:
        filename = datadir/'frame_category_2.pickle'
    else:
        filename = datadir/'frame_category_small.pickle'
    print('Data file: {}'.format(filename.as_posix()))
    with filename.open('rb') as f:
        obj = pickle.load(f)
    frames = obj['frames'][:,np.newaxis,...].astype(np.float)
    true_labels = obj['frameinfo']['category'].to_numpy()
    del(obj)
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
    net = CNN(**model_kwargs)
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

        # reset (randomize) model parameters
        for mod in net.modules():
            if hasattr(mod, 'reset_parameters'):
                print('  Resetting params for module:', mod)
                mod.reset_parameters()

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
                        del(test_outputs)
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
        del(train_set, train_loader, test_frames)
        print('  End training in epoch', final_epoch[i])

        # validate training
        validate_frames = torch.from_numpy(frames[ivalidate,...]).to(device)
        validate_labels = torch.from_numpy(true_labels[ivalidate]).to(device)
        with torch.no_grad():
            validate_outputs = net(validate_frames)
            del(validate_frames)
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
    train(data=1, repeat=1, epochs=1)
