from concurrent.futures import process
from email import header
from http import server
import numpy as np
import sys
import logging
import time
import argparse
import json
import csv
import os
import torch
import subprocess
import re
from pathlib import Path
from torch.serialization import load
from utils.combiner import Combiner
from utils.yohobuffer import YOHOBuffer
from simpleemu.simpleudp import simpleudp
from simpleemu.simplecoin import SimpleCOIN
from utils.nnetworkutils import MODEL_PATH_LIST, get_computenum, load_model, index_dict_3, load_model_mlp
from utils.packetutils import *
from utils.spliter_combiner import Spliter_combiner
from utils.mobilenet_part_split import Part_Conv_3, Part_FC_3
from utils.yoholog import *

# Network
serverAddressPort = ("10.0.0.15", 9999)
clientAddressPort = ("10.0.0.12", 9999)
ifce_name, node_ip = simpleudp.get_local_ifce_ip('10.0.')

# Model
log_cmd().info("load model...")


# Init
metadata = {}
data_list = []
num = 0
epochs = 0
epochs_index = 0
mode = 0
n_split = 0
n_combiner = 0
sc_list = [] # for split 
test_id = 0
chunk_gap = 0.0
ratio_last = 0
finished_id = 0
combiner_all = None
combiner_index = 0
combiner_list = []

# static chunk info
num_combiner = 0
num_fc = 0
cache_time = 0.0

# Simple coin
app = SimpleCOIN(ifce_name=ifce_name, n_func_process=1, lightweight_mode=True)

@app.main()
def main(simplecoin: SimpleCOIN.IPC, af_packet: bytes):
    global metadata, num, epochs_index, data_list, sc_list, ratio_last
    global n_split, n_combiner, epochs, chunk_gap, test_id, finished_id, model
    global combiner_all, num_fc, num_combiner, cache_time, cache_fc_time
    dict = index_dict_3
    packet = simpleudp.parse_af_packet(af_packet)
    if packet['Protocol'] == 17 and packet['IP_src'] != node_ip:
        chunk = packet['Chunk']
        header = int(chunk[0])
        if header == HEADER_CLEAR_CACHE:
            log_file(ifce_name).debug(f'recv clearing cache!')
            simplecoin.submit_func(pid=0, id='clear_cache')
            simplecoin.forward(af_packet)
        elif header == HEADER_INIT:
            log_file(ifce_name).debug(f'recv metadata!')
            metadata.update(pickle.loads(chunk[1:]))
            cache_time = time.time()                   # reset cache_time between epoch.
            num = get_computenum(metadata, ifce_name)  # add
            simplecoin.submit_func(pid=0, id='init_setting', args = (metadata,))
            
        elif header == HEADER_DATA or header == HEADER_FINISH:
            num_fc += 1
            if header == HEADER_DATA:
                log_file(ifce_name).debug(f'recv final fc_layer header: DATA!')
            else:
                log_file(ifce_name).debug(f'recv final fc_layer header: FINISH, nums of fc chunks total = {num_fc}')
                num_fc = 0
            simplecoin.forward(af_packet)

        elif header == HEADER_COMBINER_DATA:
            log_file(ifce_name).debug(f'recv combinered header: DATA')
            if num_combiner == 0 : # first chunk
                cache_time = time.time()
            num_combiner += 1
            if num == 0:
                simplecoin.forward(af_packet)
            #else:
                #simplecoin.submit_func(pid=0, id = 'combiner_data', args = (chunk,))
            
        elif header == HEADER_COMBINER_FINISH:
            if num_combiner == 0: # first chunk
                cache_time = time.time()
            num_combiner += 1
            log_file(ifce_name).debug(f'recv combiner header: FINISH, nums of combiners chunks total= {num_combiner}')
            if num == 0:
                simplecoin.forward(af_packet)
            else:
                cache_time = time.time() - cache_time 
                simplecoin.submit_func(pid=0, id= 'compute_forward', args=(chunk, num_combiner, cache_time))
            num_combiner = 0
            cache_time = time.time()
        
@app.func('clear_cache')
def clear_cache(simplecoin: SimpleCOIN.IPC):
    global metadata, data_list, num, epochs, epochs_index, n_split, n_combiner, sc_list, test_id, mode
    global chunk_gap, ratio_last, finished_id, combiner_all, combiner_index, combiner_list
    global num_combiner, num_fc, cache_time, cache_fc_time
    metadata = {}
    data_list = []
    num = 0
    epochs = 0
    # epochs_index = 0
    n_split = 0
    mode = 0
    n_combiner = 0
    sc_list = []
    test_id = 0
    chunk_gap = 0.0
    ratio_last = 0.0
    finished_id = 0
    combiner_all = None
    combiner_index = 0
    combiner_list = []
    num_combiner = 0
    num_fc = 0
    cache_time = 0.0

@app.func('init_setting')
def init_setting(simplecoin: SimpleCOIN.IPC, metadata2):
    global epochs_index, epochs, chunk_gap, test_id, ratio_last, num, finished_id, mode
    global n_split, n_combiner, data_list, sc_list, ifce_name, metadata, cache_time
    metadata = metadata2
    epochs_index = (epochs_index + 1) % metadata['epochs']
    if epochs_index == 0 : epochs_index = metadata['epochs']
    epochs = metadata['epochs']
    mode = metadata['mode']
    chunk_gap = metadata['chunk_gap']
    test_id = metadata['test_id']
    ratio_last = metadata['ratio_last']
    num = get_computenum(metadata, ifce_name)
    finished_id = metadata['finished_id']

    # for different vnf!
    vnf_num = index_dict_3[ifce_name] - 1
    if vnf_num == 0:
        n_split = metadata['n_split_vnf0']
        n_combiner = metadata['n_combiner_vnf0']    
    elif vnf_num == 1:
        n_split = metadata['n_split_vnf1']
        n_combiner = metadata['n_combiner_vnf1']
    elif vnf_num == 2:
        n_split = metadata['n_split_vnf2']
        n_combiner = metadata['n_combiner_vnf2']
    
    data_list = [YOHOBuffer()] * ratio_last
    sc_list = [None] * ratio_last
    # combiner_all = Combiner(None,-1,-1, n_combiner, n_combiner)
    # update metadata['finished_id'], metadata['ratio_last']
    metadata['finished_id'] += num
    metadata['ratio_last'] = ratio_last * n_split // n_combiner
    log_cmd().info('send metadata')
    simpleudp.sendto(chunk_handler.get_chunks_metadata(metadata), serverAddressPort)


    log_cmd().info("".center(60,"*"))
    log_cmd().info(f" [epochs : {epochs_index}/{epochs}] ".center(60,"*")) 
    log_cmd().info(f" [mode : {mode}] ".center(60,"*"))
    log_cmd().info(f" [n_spliter, n_combiner : {n_split, n_combiner}] ".center(60,"*"))
    log_cmd().info(f" [test_id: {test_id}] [chunk_gap = {chunk_gap}] [ratio_last = {ratio_last}] ".center(60,"*"))
    log_cmd().info("".center(60,"*")) 
    log_file(ifce_name).info("".center(60,"*"))
    log_file(ifce_name).info(f" [epochs : {epochs_index}/{epochs}] ".center(60,"*"))
    log_file(ifce_name).info(f" [mode : {mode}] ".center(60,"*"))
    log_file(ifce_name).info(f" [n_spliter, n_combiner : {n_split, n_combiner}] ".center(60,"*"))
    log_file(ifce_name).info(f" [test_id: {test_id}] [chunk_gap = {chunk_gap}] [ratio_last = {ratio_last}] ".center(60,"*"))
    log_file(ifce_name).info("".center(60,"*")) 


@app.func('combiner_data')
def combiner_data(simplecoin:SimpleCOIN.IPC, chunk):
    global data_list
    index = int(chunk[1])
    data_list[index].put(chunk[2:])


@app.func('compute_forward')
def compute_forward(simplecoin: SimpleCOIN.IPC, chunk, used_combiner_chunk, cache_time):
    global data_list, num, epochs, epochs_index, n_split, n_combiner, sc_list, test_id
    global chunk_gap, ratio_last, finished_id, metadata, mode
    global combiner_index, combiner_list, combiner_all, model_length
    #------------------------------------------------------------------------
    # time point 3 in vnf, packet arrival in vnf. We assume header resolve in switch
    print("recev the message")
    tp3_arrivalVNF = time.time() 

    weights_path = 'models_trained/best.pt'
    data_path = 'dataset/fruits/data.yaml'
    command = [
        'python3.8', 'val.py',
        '--weights', weights_path,
        '--data', data_path,
        '--save-txt',
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    tp4_sendtoSwitch = time.time()  # time point 4 in vnf which don't FC


    #------------------------------------------------------------------------
    # Regular expression to find the path
    match = re.search(r"(\d+) labels saved to (\S+)", result.stderr)
    # 使用正则表达式提取时间数据
    pre_process_time = float(re.search(r'(\d+\.\d+)ms pre-process', result.stderr).group(1))
    inference_time = float(re.search(r'(\d+\.\d+)ms inference', result.stderr).group(1))
    nms_time = float(re.search(r'(\d+\.\d+)ms NMS', result.stderr).group(1))

    # 计算单张图像的总处理时间
    process_time_per_image = pre_process_time + inference_time + nms_time
    process_time = process_time_per_image * 6

    saved_path = match.group(2)        # 'runs/val/exp8/labels'
    txt_outputs = []
    labels_dir = Path(saved_path)
    for label_file in labels_dir.glob('*.txt'):
        with open(label_file, 'rb') as file:
            txt_outputs.append(file.read())

    send_list = []
    send_list = chunk_handler.get_serialize_imagefile(
                HEADER_DATA, txt_outputs)

    #------------------------------------------------------------------------
    # send to next
    for j in range(len(send_list)):

        if j == 0:
            tp4_sendtoSwitch = time.time()  # time point 4 in vnf which don't FC
        time.sleep(chunk_gap)
        simpleudp.sendto(send_list[j], serverAddressPort)
        packet = simpleudp.parse_af_packet(send_list[j])
        return_header = int(packet['Chunk'][0])
        print(return_header)
    print("Finish process sent to next")
    #------------------------------------------------------------------------
    log_csv_tp(ifce_name,test_id, tp3_arrivalVNF, tp4_sendtoSwitch, process_time)




'''
@app.func('compute_forward')
def compute_forward(simplecoin: SimpleCOIN.IPC, chunk, used_combiner_chunk, cache_time):
    global data_list, num, epochs, epochs_index, n_split, n_combiner, sc_list, test_id
    global chunk_gap, ratio_last, finished_id, metadata, mode
    global combiner_index, combiner_list, combiner_all, model_length

    # time point 3 in vnf, packet arrival in vnf. We assume header resolve in switch
    tp3_arrivalVNF = time.time()  
    # for proc info 
    used_chunk_num = used_combiner_chunk
    used_data_len = 0
    used_data_shape = 0

    process_time = time.time()
    index = int(chunk[1])
    input_bytes = data_list[index].atomic_put_last(chunk[2:])
    if combiner_all == None: 
        combiner_all = Combiner(None,-1,-1, n_combiner, n_combiner, mode)
    
    log_file(ifce_name).debug(f"start split & combiner {index}/{len(sc_list)}, bytes_len = {len(chunk[2:])}")
    data_tensor = chunk_handler.derialize_with_index(input_bytes)
    
    input_size = data_tensor.shape[2]
    if mode == 1: 
        data_tensor.squeeze() # it should be in [1,c], not [1,1,c]
        log_file(ifce_name).debug(f"[mode: 1],change tensor data shape: {data_tensor.shape}")
    log_file(ifce_name).debug(f"derialize data to tensor, nums_float = {input_size}")

    sc_list[index] = Spliter_combiner(None, input_size, n_split, n_combiner, -1, -1, n_combiner, mode)

    # split
    log_file(ifce_name).debug(f'start split, n_split = {n_split}')
    sc_list[index].split_list = sc_list[index].spliter.split(data_tensor)

    # process
    log_file(ifce_name).debug(f'start processing, num of model_running = {num}')
    for j in range(num):
        t_start = time.time()
        log_cmd().info(f"running model {j + finished_id}, index = {index}!")
        for i in range(len(sc_list[index].split_list)):
            if j > 0: used_chunk_num = 0
            if mode == 0:
                # change dimention
                temp = sc_list[index].split_list[i]
                used_data_len = temp.shape[2] * temp.shape[1] * temp.shape[0]
                if sc_list[index].split_list[i].shape[1] == 1: 
                    dim_current = model_length[j+finished_id]
                    len_cur = sc_list[index].split_list[i].shape[2]
                    print(len_cur)
                    temp = torch.reshape(sc_list[index].split_list[i],(1,dim_current,len_cur//dim_current))
                used_data_shape = [temp.shape[0],temp.shape[1], temp.shape[2]]
                log_file(ifce_name).debug(f"reshape to shape = {temp.shape}")
                #print("----------VNF got the data size is ", sc_list[index].split_list[i].shape)
                sc_list[index].split_list[i] = model[j+finished_id](temp) 
                #print("----------the model in VNF got the data size is : ", temp.size())
            elif mode == 1:
                temp = sc_list[index].split_list[i] # data in this split
                used_data_len = temp.shape[2] * temp.shape[1] * temp.shape[0]
                used_data_shape = [used_data_len/model_length_mlp[j+finished_id], model_length_mlp[j+finished_id]]
                sc_list[index].split_list[i] = model_mlp[j+finished_id](temp)

    log_file(ifce_name).debug(f'finished process index, cost time = {time.time() - t_start} s!')
    # finished_id = num + finished_id

    # combiner [only work for serialize process!!!]
    log_file(ifce_name).debug(f'serialize combiner index = {combiner_index}, n_combiner = {n_combiner}')
    for i in range(n_split):
        combiner_data = combiner_all.combine_easy(sc_list[index].split_list[i])
        if combiner_data != None:
            log_cmd().info(f"combine [{combiner_index + 1} / {ratio_last* n_split //n_combiner}]")
            log_file(ifce_name).debug(f"combine  [{combiner_index + 1} / {ratio_last* n_split //n_combiner}]")
            if metadata['finished_id'] == 3:
                if mode == 0:
                    log_cmd().info('put to combiner_list')
                    combiner_list.append(combiner_data)
                    if combiner_index == ratio_last * n_split //n_combiner - 1:
                        # calculate fc
                        t_start = time.time()
                        log_file(ifce_name).debug(f"start process fc layer!")
                        out = torch.cat(combiner_list, dim = 2)
                        #used_chunk_num = 0
                        #used_data_shape = [out.shape[0],out.shape[1],out.shape[2]]
                        #used_data_len = out.shape[0] * out.shape[1] * out.shape[2]
                        fc_data = model[3](out)
                        log_file(ifce_name).debug(f"finished process fc, cost time = {time.time() - t_start} s!")
                        # log_csv(ifce_name, test_id, chunk_gap, epochs_index, epochs, n_split * ratio_last, n_combiner, used_chunk_num,
                        #         used_data_len, used_data_shape, cache_time_fc, time.time()- t_start + process_time, True)
                        chunk_arr = chunk_handler.get_chunks_fc(fc_data)
                        log_file(ifce_name).debug(f'send {len(chunk_arr)} to server! ')
                        time.sleep(chunk_gap)
                        tp4_sendtoSwitch = time.time()  # time point 4 in vnf which have do the FC
                        for chunk in chunk_arr:
                            time.sleep(chunk_gap)
                            simpleudp.sendto(chunk,serverAddressPort)
                        log_file(ifce_name).debug(f'send HEADER_FINISH flag to server!')
                        combiner_all = None
                        combiner_index = 0
                elif mode == 1:
                    log_cmd().info('combiner part finished !')
                    combiner_list.append(combiner_data)
                    chunk_arr = chunk_handler.get_serialize_torcharray(HEADER_DATA, None, combiner_data)
                    if combiner_index != ratio_last * n_split //n_combiner - 1:
                        chunk_arr[-1] = bytes([HEADER_DATA]) + chunk_arr[-1][1:]
                    else:
                        combiner_all = None
                        combiner_index = 0
                    log_file(ifce_name).debug(f'send {len(chunk_arr)} to server! ')
                    time.sleep(chunk_gap)
                    for chunk in chunk_arr:
                        time.sleep(chunk_gap)
                        simpleudp.sendto(chunk,serverAddressPort)
                    # log_file(ifce_name).debug(f'send HEADER_FINISH flag to server!')


            else:
                # send to next
                chunk_arr = chunk_handler.get_serialize_torcharray(HEADER_COMBINER_DATA, combiner_index, combiner_data)
                #print("----------VNF send the data size is : ", combiner_data.size())
                log_cmd().info(f'send combiner data, len={len(chunk_arr)}')
                log_file(ifce_name).debug(f'[part_index ={index} , combiner_index={combiner_index}, send {len(chunk_arr)} chunks to server!')
                time.sleep(chunk_gap)
                for j in range(len(chunk_arr)):
                    if j == 0:
                        tp4_sendtoSwitch = time.time()  # time point 4 in vnf which don't FC
                    time.sleep(chunk_gap)
                    simpleudp.sendto(chunk_arr[j], serverAddressPort)
            combiner_index += 1
            if combiner_index == ratio_last * n_split // n_combiner:
                combiner_index = 0
                combiner_all = None

    process_time = time.time() - process_time
    log_csv(ifce_name, mode, test_id, chunk_gap, epochs_index, epochs, n_split * ratio_last, n_combiner, used_chunk_num,
                    used_data_len, used_data_shape, cache_time, process_time)
    log_csv_tp(ifce_name,test_id, tp3_arrivalVNF, tp4_sendtoSwitch)
'''
@app.func('test')
def test(simplecoin: SimpleCOIN.IPC):
    global model
    log_file(ifce_name).debug('test speed!')
    log_cmd().info(f'test_speed,{len(model)}')


app.run()