from cgi import test
from http import server
import sys
import numpy as np
import logging
import time
import argparse
import json
import csv
import os
import torch
from torch.serialization import load
from simpleemu.simpleudp import simpleudp
from simpleemu.simplecoin import SimpleCOIN
from utils.nnetworkutils import *
from utils.packetutils import *
from utils.spliter_combiner import Spliter_combiner
from utils.mobilenet_part_split import Part_Conv_3, Part_FC_3
from utils.yoholog import *

# metadata
metadata = {}
mode = 0
finished_id = 0
epochs = 1
test_id = 1
chunk_gap = CHUNK_GAP
n_split_client = 128
n_split_vnf0 = 1
n_split_vnf1 = 1
n_split_vnf2 = 1
n_split_server = 1
n_combiner_client = 1
n_combiner_vnf0 = 4
n_combiner_vnf1 = 8
n_combiner_vnf2 = 1
n_combiner_server = 1


def init_metadata():
    global epochs, test_id, metadata, MTU, is_split, is_debug, id, length
    DEF_METADATA = {
        'mode': mode,
        'finished_id': 0,
        'test_id': test_id,
        'epochs': epochs,
        'chunk_gap': chunk_gap,
        'n_split_client': n_split_client,
        'n_split_vnf0': n_split_vnf0,
        'n_split_vnf1': n_split_vnf1,
        'n_split_vnf2': n_split_vnf2,
        'n_split_server': n_split_server,
        'n_combiner_client': n_combiner_client,
        'n_combiner_vnf0': n_combiner_vnf0,
        'n_combiner_vnf1': n_combiner_vnf1,
        'n_combiner_vnf2': n_combiner_vnf2,
        'n_combiner_server': n_combiner_server,
        'ratio_last': n_split_client//n_combiner_client
    }
    metadata.update(DEF_METADATA)


# network setting
ifce_name, node_ip = simpleudp.get_local_ifce_ip('10.0')
log_file(ifce_name).debug(f"ifce_name = {ifce_name}, node_ip = {node_ip}")
app = SimpleCOIN(ifce_name=ifce_name, n_func_process=5, lightweight_mode=True)
serverAddressPort = ("10.0.0.15", 9999)
clientAddressPort = ("10.0.0.12", 9999)

# setting args
def set_args(args):
    global epochs, test_id, chunk_gap, mode
    global n_split_client, n_split_vnf0, n_split_vnf1, n_split_vnf2, n_split_server
    global n_combiner_client, n_combiner_vnf0, n_combiner_vnf1, n_combiner_vnf2, n_combiner_server

    mode=args.mode
    epochs = args.epochs
    test_id = args.test_id
    chunk_gap = args.chunk_gap
    n_split_client = args.n_split_client
    n_split_vnf0 = args.n_split_vnf0
    n_split_vnf1 = args.n_split_vnf1
    n_split_vnf2 = args.n_split_vnf2
    n_split_server = args.n_split_server
    n_combiner_client = args.n_combiner_client
    n_combiner_vnf0 = args.n_combiner_vnf0
    n_combiner_vnf1 = args.n_combiner_vnf1
    n_combiner_vnf2 = args.n_combiner_vnf2
    n_combiner_server = args.n_combiner_server





# genarate data
log_cmd().info("gearate data ...")
#data = get_network_data_tensors()            # .numpy() # torch.from_numpy(c)

data = get_yolo_data_tensors()
length = len(data)


# others
epochs_index = 1
'''
[Debug]
    processing_time   :   Data calculate time in client.
    serialize_time    :   serialize send data from torch.array -> bytes
    packet_send_time  :   send all the chunk to next node
    total_time        :   all the time in each epoch

[Release & Debug]
    transmission_time :   send first chunk -> service ack recv all
    service_time      :   send first chunk -> service ack finish calculate
'''
LATENCY = {"processing_time": 0.0, "serialize_time": 0.0, "packet_send_time": 0.0,
           "transmission_time": 0.0, "service_time": 0.0, "total_time": 0.0}
latency = {}
latency.update(LATENCY)

# Simple coin
app = SimpleCOIN(ifce_name=ifce_name, n_func_process=1, lightweight_mode=True)
# run 1 epoch
def run():
    global metadata, model, mode, data, data_shape
    global epochs_index, latency, finished_id, data_mlp, data_shape_mlp
    # for proc info var
    used_chunk_num = 0
    used_data_len = 0
    used_data_shape = None

    # init
    init_metadata()
    latency = {}
    latency.update(LATENCY)

    log_cmd().info("".center(60, "*"))
    log_cmd().info(f" [epochs : {epochs_index}/{epochs}] ".center(60, "*"))
    log_cmd().info(f" [mode : {mode}]".center(60, "*"))
    log_cmd().info(
        f" [n_spliter, n_combiner : {n_split_client, n_combiner_client}] ".center(60, "*"))
    log_cmd().info(
        f" [test_id: {test_id}] [chunk_gap = {chunk_gap} ]".center(60, "*"))
    log_cmd().info("".center(60, "*"))

    log_file(ifce_name).debug("".center(60, "*"))
    log_file(ifce_name).debug(
        f" [epochs : {epochs_index}/{epochs}] ".center(60, "*"))
    log_file(ifce_name).debug(f"[mode : {mode}]".center(60, "*"))
    log_file(ifce_name).debug(
        f" [n_spliter, n_combiner : {n_split_client, n_combiner_client}] ".center(60, "*"))
    log_file(ifce_name).debug(
        f" [test_id: {test_id}] [chunk_gap = {chunk_gap} ] [ratio_last = {metadata['ratio_last']}]".center(60, "*"))
    log_file(ifce_name).debug("".center(60, "*"))

    # start
    t_start_service = time.time()

    compute_num = get_computenum(metadata, ifce_name)
    finished_id = metadata['finished_id']
    metadata['finished_id'] += compute_num

    log_cmd().info('send clear cache command')
    simpleudp.sendto(chunk_handler.get_chunks_clean(), serverAddressPort)

    log_cmd().info('send metadata')
    simpleudp.sendto(chunk_handler.get_chunks_metadata(
        metadata), serverAddressPort)

    # model is for computer, if u just want use spliter & combiner basic function. don't need model
    if mode == 0:
        LENGTH = length
    elif mode == 1:
        LENGTH = length_mlp
    sc = Spliter_combiner(None,
                          LENGTH, n_split_client, n_combiner_client, -1, -1, n_combiner_client, mode)
    # split
    if mode == 0:
        sc.split_list = sc.spliter.split(data)
    elif mode == 1:
        sc.split_list = sc.spliter.split(data_mlp)
        print("split data shape:", sc.split_list[0].shape)
    # process
    t_start = time.time()
    for j in range(compute_num):
        log_cmd().info(f"running model {j}!")
        for i in range(len(sc.split_list)):
            used_data_shape = sc.split_list[i].shape
            used_data_len = used_data_shape[2]
            cache_time = 0.0
            if mode == 0:
                sc.split_list[i] = model[j](sc.split_list[i])
            elif mode == 1:
                sc.split_list[i] = model_mlp[j](sc.split_list[i])
            log_csv(ifce_name, test_id, chunk_gap, epochs_index, epochs, 0, n_split_client, n_combiner_client,
                    used_chunk_num, used_data_len, used_data_shape, cache_time, time.time()-t_start)
        finished_id += 1
    latency["processing_time"] += time.time() - t_start

    # combine
    #sc.combine_split_list()

    # check if all finished
    t_start = time.time()
    if finished_id == 3:
        out = None # finished data
        if mode == 0:
            log_cmd().info("all finished! start fc layer!")
            out = torch.cat(sc.combiner_list, dim=2)
            used_data_shape = out.shape
            used_data_len = used_data_shape[0] * used_data_shape[1] * used_data_shape[2]
            out = model[3](out)
            latency["processing_time"] += time.time() - t_start
            log_csv(ifce_name, test_id, chunk_gap, epochs_index, epochs, n_split_client, n_combiner_client,
                        used_chunk_num, used_data_len, used_data_shape, time.time()-t_start)
        elif mode == 1:
            log_cmd().info("all finished!")
            out = torch.cat(sc.combiner_list, dim=0)
        
        # send finished data
        chunk_arr = chunk_handler.get_chunks_fc(out)
        log_file(ifce_name).debug(f'send {len(chunk_arr)} to server! ')
        time.sleep(chunk_gap)
        for chunk in chunk_arr:
            time.sleep(chunk_gap)
            simpleudp.sendto(chunk, serverAddressPort)
        log_file(ifce_name).debug(f'send HEADER_FINISH flag to server!')
    else:
        send_list = []
        send_list = chunk_handler.get_serialize_imagefile(
                HEADER_COMBINER_DATA, data)
        for i, chunk in enumerate(send_list):
            time.sleep(chunk_gap)
            if i == 0:
                tp_clientsentPacket = time.time()
            simpleudp.sendto(chunk, serverAddressPort)
        log_cmd().info(f"Finish: send images to server! Used time is {time.time() - tp_clientsentPacket}")
        '''
        # yolo send data to the vnf, need to add head to each chuncks.
        log_cmd().info(
            f"finish: {metadata['finished_id']} / 3, send combiner_list to server!")
        send_list = [None] * len(sc.combiner_list)
        for i in range(len(sc.combiner_list)):
            send_list[i] = yolo_chunk_handle.get_serialize_torcharray(
                HEADER_COMBINER_DATA, i, sc.combiner_list[i])
            log_file(ifce_name).debug(
                f'[index = {i}, send {len(send_list[i])} to server!')
            time.sleep(chunk_gap)
            for j in range(len(send_list[i])):
                time.sleep(chunk_gap)
                if j == 0 and i == 0:
                    tp_clientsentPacket = time.time()
                simpleudp.sendto(send_list[i][j], serverAddressPort)
        log_file(ifce_name).debug(
            f'[index = {i}, send HEADER_COMBINER_FINISH flag to server!')
        '''
    latency['serialize_time'] += time.time() - t_start

    # wait ack from server
    log_cmd().info("*** wait ack from server !")

    count = 0
    return_header = 0
    while True:
        dst_addr, pred_data = simpleudp.recvfrom(1000)
        if test_id == 2:
            count = count + 1
            if count == 12:
                return_header = HEADER_FINISH
        else:
            packet = simpleudp.parse_af_packet(dst_addr)
            return_header = int(packet['Chunk'][0])

        if return_header == HEADER_FINISH:
            break
    tp_clientGetPacket = time.time()    
    log_cmd().info("*** get the result from server !")
    log_csv_tp(ifce_name,test_id,tp_clientsentPacket, tp_clientGetPacket)

def run_one_epochs():
    global latency, epochs, chunk_gap, data_shape, epochs_index

    # save all latency in debug mode

    start_time = time.time()
    log_file(ifce_name).debug(f" start epochs: {epochs_index}/{epochs} ")
    run()
    end_time = time.time()
    log_file(ifce_name).debug(f" end epochs: {epochs_index}/{epochs}")
    latency["total_time"] = end_time - start_time
    print("*** write result in client !")
    filename = "./measurements/testcase_" + \
        str(test_id)+"/client_all.txt"
    with open(filename, 'a') as f:
        f.write(json.dumps(latency) + ",\n")

    # save transmission time, service time in csv file.

    csvname = "./measurements/testcase_" + \
        str(test_id)+"/client.csv"
    with open(csvname, "a+") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        if os.path.getsize(csvname) == 0:
            writer.writerow(
                ["epochs", "current_epochs", "test_id", "mode", "chunk_gap",
                    "n_split_client", "n_split_vnf0", "n_split_vnf1", "n_split_vnf2", "n_split_server",
                    "n_combiner_client", "n_combiner_vnf0", "n_combiner_vnf1", "n_combiner_vnf2", "n_combiner_server",
                    "transmision_time", "service_time"])
        writer.writerow(
            [epochs, epochs_index, test_id, mode, chunk_gap,
                n_split_client, n_split_vnf0, n_split_vnf1, n_split_vnf2, n_split_server,
                n_combiner_client, n_combiner_vnf0, n_combiner_vnf1, n_combiner_vnf2, n_combiner_server,
                round(latency["transmission_time"], 3), round(latency["service_time"], 4)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yoho Emulator: Client")
    
    # --mode = 0
    parser.add_argument(
        "--mode",
        type=int,
        default=0,
        help="0 for YOHO and 1 for MLP"
    )
    # --epochs = 5
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="The number of validation epochs. default = 5"
    )

    '''
    compute_table = np.array([
        [3, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 3],
        [0, 3, 0, 0, 0],
        [0, 0, 3, 0, 0],
        [0, 0, 0, 3, 0]
    ])
    '''

    # --testid = 1
    parser.add_argument(
        "--test_id",
        type=int,
        default=5,
        choices=[0, 1, 2, 3, 4, 5],  # see the setting on nnetworkutils.py
        help="ID of the test case. default = 1"
    )

    # chunk_gap = 0.001
    parser.add_argument(
        "--chunk_gap",
        type=float,
        default=0.002,
        help="dalay for send each chunk [s]. default = 0.001"   # [0.001-0.01]
    )

    # n_spliter_client = 8
    parser.add_argument(
        "--n_split_client",
        type=int,
        default=1,
        help="number of split for client. default = 8"
    )

    # n_spliter_vnf0 = 1
    parser.add_argument(
        "--n_split_vnf0",
        type=int,
        default=1,
        help="number of split for vnf0. default = 1"
    )

    # n_spliter_vnf1 = 1
    parser.add_argument(
        "--n_split_vnf1",
        type=int,
        default=1,
        help="number of split for vnf1. default = 1"
    )

    # n_spliter_vnf2 = 1
    parser.add_argument(
        "--n_split_vnf2",
        type=int,
        default=1,
        help="number of split for vnf2. default = 1"
    )

    # n_spliter_server = 1
    parser.add_argument(
        "--n_split_server",
        type=int,
        default=1,
        help="number of split for server. default = 1"
    )

    # n_combiner_client = 1
    parser.add_argument(
        "--n_combiner_client",
        type=int,
        default=1,
        help="number of combiner for client. default = 1"
    )

    # n_combiner_vnf0 = 1
    parser.add_argument(
        "--n_combiner_vnf0",
        type=int,
        default=1,
        help="number of combiner for vnf0. defalut = 1"
    )

    # n_combiner_vnf1 = 1
    parser.add_argument(
        "--n_combiner_vnf1",
        type=int,
        default=1,
        help="number of combiner for vnf1. default = 1"
    )

    # n_combiner_vnf2 = 1
    parser.add_argument(
        "--n_combiner_vnf2",
        type=int,
        default=1,
        help="number of combiner for vnf2. default = 1"
    )
    # n_combiner_server = 1
    parser.add_argument(
        "--n_combiner_server",
        type=int,
        default=1,
        help="number of combiner for server. default = 1"
    )

    parser.set_defaults(func=set_args)
    args = parser.parse_args()
    if args.test_id == 3 or args.test_id == 2:
        args.n_split_client = 1
        args.n_split_vnf0 = 1
        args.n_split_vnf1 = 1
        args.n_split_vnf2 = 1

        args.n_combiner_vnf0 = 1
        args.n_combiner_vnf1 = 1
        args.n_combiner_vnf2 = 1
        args.n_combiner_server = 1

    args.func(args)

    for i in range(epochs):
        log_cmd().info("start run!")
        info1 = f" [client]: n_spliter:{n_split_client}, n_combiner:{n_combiner_client} "
        info2 = f" [vnf0]: n_spliter:{n_split_vnf0}, n_combiner:{n_combiner_vnf0} "
        info3 = f" [vnf1]: n_spliter:{n_split_vnf1}, n_combiner:{n_combiner_vnf1} "
        info4 = f" [vnf2]: n_spliter:{n_split_vnf2}, n_combiner:{n_combiner_vnf2} "
        info5 = f" [server]: n_spliter:{n_split_server}, n_combiner:{n_combiner_server} "
        log_cmd().info("".center(60, "*"))
        log_cmd().info(info1.center(60, "*"))
        log_cmd().info(info2.center(60, "*"))
        log_cmd().info(info3.center(60, "*"))
        log_cmd().info(info4.center(60, "*"))
        log_cmd().info(info5.center(60, "*"))
        log_cmd().info("".center(60, "*"))
        log_file(ifce_name).debug("".center(60, "*"))
        log_file(ifce_name).debug(info1.center(60, "*"))
        log_file(ifce_name).debug(info2.center(60, "*"))
        log_file(ifce_name).debug(info3.center(60, "*"))
        log_file(ifce_name).debug(info4.center(60, "*"))
        log_file(ifce_name).debug(info5.center(60, "*"))
        log_file(ifce_name).debug("".center(60, "*"))
        time.sleep(20)
        run_one_epochs()
        epochs_index += 1
