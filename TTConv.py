# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:53:07 2019

@author: tob10
"""

class TTConv(torch.nn.Module):
    def __init__(inp,         
                 conv_size,
                 inp_ch_modes,              
                 out_ch_modes,
                 ranks,
                 strides=[1, 1],
                 padding=[0,0],
                 trainable=True):
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
    """       
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TTConv, self).__init__()
        
        # filter initialiased with glorot initialisation with the right parameter
        self.filters = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(conv_size[0], conv_size[1], 1, ranks[0])))
        d = inp_ch_modes.size
        
        self.cores = nn.ParameterList()
        for i in range(d):
            # initialise each core with once again glorot initialisation with parameter matching the output and input channel mode (the c_i/s_i multiply to C/S)
            self.cores.append(nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(out_ch_modes[i] * ranks[i + 1], ranks[i] * inp_ch_modes[i]))))
            


    def forward(self, x):
        
        # should we use clone to keep self.filters and self.cores untouched? 
        full = self.filters
        cores = self.cores
        
        
        for i in range(d):            
            full = full.view([-1, ranks[i]])
            core = torch.transpose(cores[i], 1, 0])
            core = core.view([ranks[i], -1])
            full = torch.mm(full, core)
            
        out_ch = np.prod(out_ch_modes)
        
        fshape = [conv_size[0], conv_size[1]]
        order = [0, 1]
        inord = []
        outord = []
        for i in range(d):
            fshape.append(inp_ch_modes[i])
            inord.append(2 + 2 * i)
            fshape.append(out_ch_modes[i])
            outord.append(2 + 2 * i + 1)
        order += inord + outord
        
        #check syntax here
        full = full.view(fshape)
        full = torch.transpose(full, order)
        full = full.view([window[0], window[1], inp_ch, out_ch])
        
        
        inp_shape = inp.get_shape().as_list()[1:]
        inp_h, inp_w, inp_ch = inp_shape[0:3]
        tmp = tf.reshape(inp, [-1, inp_h, inp_w, inp_ch])
        tmp = tf.nn.conv2d(tmp,
                           full,
                           [1] + strides + [1],
                           padding,
                           name='conv2d')
        
        
        
        
        return tmp