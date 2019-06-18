import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TTDeconv_full(torch.nn.Module):
    def __init__(self,
                 conv_size,
                 inp_ch_modes,
                 out_ch_modes,
                 ranks,
                 stride=1,
                 padding=0):
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

        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TTDeconv_full, self).__init__()
        self.conv_size = conv_size
        self.inp_ch_modes = inp_ch_modes
        self.out_ch_modes = out_ch_modes
        self.ranks = ranks
        self.stride = stride
        self.padding = padding

        # filter initialiased with glorot initialisation with the right parameter
        self.filters = nn.Parameter(
            torch.nn.init.xavier_uniform_(
                torch.empty(conv_size[0], conv_size[1], 1, ranks[0])
            )
        )
        self.d = len(inp_ch_modes)

        self.cores = nn.ParameterList()
        for i in range(self.d):
            # initialise each core with once again glorot initialisation with
            # parameter matching the output and input channel mode
            # (the c_i/s_i multiply to C/S)
            empty_core = torch.empty(
                out_ch_modes[i] * ranks[i + 1], ranks[i] * inp_ch_modes[i]
            )
            empty_core = nn.Parameter(
                torch.nn.init.xavier_uniform_(empty_core)
            )
            self.cores.append(empty_core)

    def forward(self, inp):
        # should we use clone to keep self.cores untouched?
        cores = self.cores
        full = self.filters
        # inp_shape = inp.get_shape().as_list()[1:] + one other line
        inp_ch, inp_h, inp_w = list(inp.shape)[1:4]  # shape 0 is batchsize
        tmp = torch.reshape(inp, (-1, inp_ch, inp_h, inp_w))

        for i in range(self.d):
            full = torch.reshape(full, (-1, self.ranks[i]))
            core = torch.transpose(cores[i], 1, 0)
            core = torch.reshape(core, (self.ranks[i], -1))
            full = torch.mm(full, core)
        out_ch = np.prod(self.out_ch_modes)

        fshape = [self.conv_size[0], self.conv_size[1]]
        order = [0, 1]
        inord = []
        outord = []
        for i in range(self.d):
            fshape.append(self.inp_ch_modes[i])
            inord.append(2 + 2 * i)
            fshape.append(self.out_ch_modes[i])
            outord.append(2 + 2 * i + 1)

        order += inord + outord
        full = torch.reshape(full, tuple(fshape))
        full = full.permute(tuple(order))
        full = torch.reshape(full, (inp_ch, out_ch, self.conv_size[0], self.conv_size[1]))
        tmp = F.conv_transpose2d(tmp, full, bias=None, stride=self.stride, padding=self.padding)

        return tmp
