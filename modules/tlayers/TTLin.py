# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:53:07 2019

@author: tob10
"""

import numpy as np
import torch
import torch.nn as nn


class TTLin(torch.nn.Module):
    def __init__(self,
                 inp_modes,              
                 out_modes,
                 mat_ranks):
        """
        tt-conv-layer (convolution of full input tensor with tt-filters
        (make tt full then use conv2d))
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
        """       
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TTLin, self).__init__()
        self.inp_modes = inp_modes
        self.out_modes = out_modes
        self.mat_ranks = mat_ranks
        # filter initialiased with glorot initialisation with the right parameter
        self.d = inp_modes.size
        
        self.mat_cores = nn.ParameterList()
        for i in range(self.d):
            empty_core = torch.empty(
                out_modes[i] * mat_ranks[i + 1], mat_ranks[i] * inp_modes[i]
            )
            empty_core = nn.Parameter(
                torch.nn.init.xavier_uniform_(empty_core)
            )
            # initialise each core with once again glorot initialisation with
            # parameter matching the output and input channel mode
            # (the c_i/s_i multiply to C/S)
            self.mat_cores.append(empty_core)

    def forward(self, inp):
        # should we use clone to keep self.cores untouched? 
        mat_cores = self.mat_cores
        out = torch.reshape(inp, (-1, np.prod(self.inp_modes)))
        out = torch.transpose(out, 1, 0)
        for i in range(self.d):
            out = torch.reshape(out, (self.mat_ranks[i] * self.inp_modes[i], -1))
            out = torch.mm(mat_cores[i], out)
            out = torch.reshape(out, (self.out_modes[i], -1))
            out = torch.transpose(out, 1, 0)
        out = torch.reshape(out, (-1, np.prod(self.out_modes)))
        
        return out
