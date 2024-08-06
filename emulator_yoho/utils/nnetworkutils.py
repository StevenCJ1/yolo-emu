#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   nnetworkutils.py
@Time    :   2022/01/04 11:55:05
@Author  :   Jiakang Weng
@Version :   1.0
@Contact :   jiakang.weng@mailbox.tu-dresden.de
@License :   (C)Copyright 2021-2022
@Desc    :   None
'''

# here put the import lib
import torch
import numpy as np
import logging


from simpleemu.simplecoin import SimpleCOIN
from simpleemu.simpleudp import simpleudp
from utils.packetutils import *
from utils.yoholog import *
from utils.mlp_part_split import *


MODEL_PATH_LIST= ["./results/distributed_models/part_1.pt",
                    "./results/distributed_models/part_2.pt",
                    "./results/distributed_models/part_3.pt"]

MODEL_PATH_LIST_mlp = ["./results/mlp/part1.pth",
                    "./results/mlp/part2.pth",
                    "./results/mlp/part3.pth"]

model = []
datas = []
dst_ip_addr = None

def load_model(model_path):
    model = torch.load(model_path, map_location="cpu")["model"].eval()
    model.mode = "val"
    return model

def load_model_mlp():
    model1 = MLP1(3072)
    model2 = MLP2()
    model3 = MLP3()
    model1.eval()
    model2.eval()
    model3.eval()
    model1.load_state_dict(torch.load(MODEL_PATH_LIST_mlp[0]))
    model2.load_state_dict(torch.load(MODEL_PATH_LIST_mlp[1]))
    model3.load_state_dict(torch.load(MODEL_PATH_LIST_mlp[2]))
    return [model1, model2, model3]

compute_table = np.array([
    [3, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 3],
    [0, 3, 0, 0, 0],
    [0, 0, 3, 0, 0],
    [0, 0, 0, 3, 0]
])


index_dict_3 = {
    "client-s0": 0,
    "vnf0-s0": 1,
    "vnf1-s1": 2,
    "vnf2-s2": 3,
    "server-s2": 4
}

logging_level_table = {
    "release": logging.INFO,
    "debug"  : logging.DEBUG
}


def get_computenum(metadata,ifce_name):
    global index_dict_3, compute_table
    test_id = metadata["test_id"]
    ifce_name = ifce_name
    # from table get the num
    len_link = len(compute_table[test_id])
    assert(len_link == 5)
    dict = index_dict_3
    node_index = dict[ifce_name]
    return compute_table[test_id][node_index]

