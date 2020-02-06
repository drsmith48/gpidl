#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self,
                 conv_layers=[5,15],
                 conv_size=[5,5],
                 fc_layers=[80,60],
                 activation='leaky_relu',
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
