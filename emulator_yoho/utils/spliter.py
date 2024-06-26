#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   spliter.py
@Time    :   2022/03/20 11:20:24
@Author  :   Jiakang Weng
@Version :   1.0
@Contact :   jiakang.weng@mailbox.tu-dresden.de
@License :   (C)Copyright 2021-2022
@Desc    :   Spliter for yoho, spliter in time space, based on ia-net, changed to use yohobuffer.
'''

# here put the import lib
import torch

class Spliter:
    def __init__(self, model, input_size, s, n, mode: 0) -> None:
        '''
        model:
        input_size: shape of input data, size should be like [b, c, t] 
        s: stride size, depends on the model
        n: number of splited data
        '''
        # input_size // n = data length in each block which should more than stride.
        # assert n <= input_size // s
        self.model = model
        self.s = s
        self.n = n
        self.size = input_size // n
        self.mode = mode
        if mode == 0: self.dim = 2
        elif mode == 1: self.dim = 0

    def compute_once(self, x):
        b, c, t = x.shape
        left = t % self.s
        if left != 0:
            zeros_pad = torch.zeros([b, c, self.s - left], dtype=torch.float32).cpu()
            x = torch.cat([x, zeros_pad], dim=2)
        return self.model(x)
    
    def split(self, x):
        if self.n == 1 or self.n == 0:  return [x]
        if self.mode == 0 : 
            if len(x.size()) == 2:
                x = x.unsqueeze(0)
            split_list = torch.split(x, self.size, dim=2)
            split_list = list(split_list)
            for i in range(len(split_list)):
                b, c, t = split_list[i].shape
                left = t % self.s
                if left != 0 :
                    zeros_pad = torch.zeros([b, c, self.s - left], dtype=torch.float32).cpu()
                    split_list[i] = torch.cat([split_list[i], zeros_pad], dim=2)
        elif self.mode == 1:
            assert len(x.size()) == 2
            if self.n == 1 or self.n == 0:  return [x]
            split_list = torch.split(x, self.size, dim=0)
            split_list = list(split_list)
        return split_list

    def split_and_compute(self, x):
        if self.n == 1:
            return self.model(x)
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        split_list = torch.split(x, self.size, dim=2)
        ans = []
        for sub_input in split_list:
            out = self.compute_once(sub_input)
            '''
            
            '''
            ans.append(out)
        return ans

