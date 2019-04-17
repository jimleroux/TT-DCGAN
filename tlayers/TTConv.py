# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:53:07 2019

@author: tob10
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TTConv(torch.nn.Module):
    def __init__(self, 
                 conv_size,
                 inp_ch_modes,              
                 out_ch_modes,
                 ranks,
                 strides=[1, 1],
                 padding=[0,0]):
        """ tt-conv-layer (convolution of full input tensor with tt-filters (make tt full then use conv2d))
        Args:
        inp: input tensor, float - [batch_size, H, W, C]
        conv_size: convolution window size, list [wH, wW]
        inp_ch_modes: input channels modes, np.array (int32) of size d
        out_ch_modes: output channels modes, np.array (int32) of size d
        ranks: tt-filters ranks, np.array (int32) of size (d + 1)        
        strides: strides, list of 2 ints - [sx, sy] 
        padding    
        trainable: trainable variables flag, bool

        Returns:
            out: output tensor, float - [batch_size, prod(out_modes)]

        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TTConv, self).__init__()
        
        self.inp_ch_modes=inp_ch_modes
        self.out_ch_modes=out_ch_modes
        self.ranks=ranks
        self.stride=tuple([1] + strides + [1])
        self.padding=tuple(padding)
        
        # filter initialiased with glorot initialisation with the right parameter
        self.filters = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(conv_size[0], conv_size[1], 1, ranks[0])))
        self.d = inp_ch_modes.size
        
        self.cores = nn.ParameterList()
        for i in range(self.d):
            # initialise each core with once again glorot initialisation with parameter matching the output and input channel mode (the c_i/s_i multiply to C/S)
            self.cores.append(nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(out_ch_modes[i] * ranks[i + 1], ranks[i] * inp_ch_modes[i]))))
            


    def forward(self, inp):
        
        # should we use clone to keep self.cores untouched? 
        cores = self.cores
        
        #inp_shape = inp.get_shape().as_list()[1:] + one other line
        inp_h, inp_w, inp_ch = list(inp.shape)[1:4] #shape 0 is batchsize
        #tmp = tf.reshape(inp, [-1, inp_h, inp_w, inp_ch])
        tmp = torch.reshape(inp, (-1, inp_h, inp_w, inp_ch))
        #tmp = tf.transpose(tmp, [0, 3, 1, 2])
        tmp = torch.transpose(tmp,3,1) # put indice 3 at one
        tmp = torch.transpose(tmp,2,3) # now 1 is at 3, so switch it with 2
        
        #tmp = tf.reshape(tmp, [-1, inp_h, inp_w, 1]) 
        tmp = torch.reshape(tmp, (-1, inp_h, inp_w, 1)) # why not using inp_h and inp_w as first and second entry?
        
        #tmp = tf.nn.conv2d(tmp, filters, [1] + strides + [1], padding)  
        tmp = F.conv2d(tmp, self.filters,bias=None, stride=self.stride, padding=self.padding)  #might need to look at the order of the stride
        
        #tmp shape = [batch_size * inp_ch, h, w, r]
        #h, w = tmp.get_shape().as_list()[1:3]
        h, w = list(tmp.shape)[1:3]
        
        #tmp = tf.reshape(tmp, [-1, inp_ch, h, w, ranks[0]])
        tmp = torch.reshape(tmp, (-1, inp_ch, h, w, self.ranks[0]))
        
        #tmp = tf.transpose(tmp, [4, 1, 0, 2, 3])        
        tmp = torch.transpose(tmp, 4, 0) #[4,1,2,3,0]
        tmp = torch.transpose(tmp, 4, 3) #[4,1,2,0,3]
        tmp = torch.transpose(tmp, 2, 3) #[4,1,0,2,3]
        #tmp shape = [r, c, b, h, w]
        
        
        for i in range(self.d):            
            #tmp = tf.reshape(tmp, [ranks[i] * inp_ch_modes[i], -1])
            tmp = torch.reshape(tmp, (self.ranks[i] * self.inp_ch_modes[i], -1))
            #tmp = tf.matmul(cores[i], tmp)
            tmp = torch.mm(cores[i], tmp)            
            #tmp = tf.reshape(tmp, [out_ch_modes[i], -1])     
            tmp = torch.reshape(tmp, (self.out_ch_modes[i], -1))
            #tmp = tf.transpose(tmp, [1, 0])
            tmp = torch.transpose(tmp, 1, 0)
        
        out_ch = np.prod(self.out_ch_modes)
        
        #out = tf.reshape(tmp, [-1, h, w, out_ch], name='out')
        out = torch.reshape(tmp, (-1, h, w, out_ch))
        
        
        return out