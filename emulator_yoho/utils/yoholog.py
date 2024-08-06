import logging
from tabnanny import filename_only
import time
import csv
import os
import pandas as pd

index_dict_3 = {
    "client-s0": 0,
    "vnf0-s0": 1,
    "vnf1-s1": 2,
    "vnf2-s2": 3,
    "server-s2": 4
}

time_str = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())


def log_cmd():
    log_cmd = logging.getLogger("console-logger")
    log_cmd.handlers.clear()
    stream_handler = logging.StreamHandler()
    log_cmd.addHandler(stream_handler)
    log_cmd.setLevel(logging.INFO)
    return log_cmd


def log_file(ifce_name: str):
    log_file = logging.getLogger("file-logger")
    log_file.handlers.clear()
    file_name = "./measurements/log/yoho_" + ifce_name + "_" + time_str + ".log"
    file_handler = logging.FileHandler(file_name)
    file_handler.setFormatter(logging.Formatter(
        "%(filename)s | %(asctime)s | %(message)s"))
    log_file.addHandler(file_handler)
    log_file.setLevel(logging.DEBUG)
    return log_file

def log_file_tp(ifce_name: str):  # log file for each time points
    log_file = logging.getLogger("file-logger")
    log_file.handlers.clear()
    file_name = "./measurements/log/yoho_" + ifce_name + "_" + time_str + '_tp' +".log"
    file_handler = logging.FileHandler(file_name)
    file_handler.setFormatter(logging.Formatter(
        "%(filename)s | %(asctime)s | %(message)s"))
    log_file.addHandler(file_handler)
    log_file.setLevel(logging.DEBUG)
    return log_file


def log_csv_tp(ifce_name, test_id: int, t0:float,t1:float=0, t2:float=0,is_update:bool=False):
    file_name = "./measurements/testcaseTP_" + \
        str(test_id) + "/tp_" + ifce_name + ".csv"

    if is_update:
        delLastLine(file_name)
    if t2 == 0:
        with open(file_name, "a+") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")

            if os.path.getsize(file_name) == 0:
                writer.writerow(["t0", "t1"])
            writer.writerow([t0,t1])
    else:
        with open(file_name, "a+") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")

            if os.path.getsize(file_name) == 0:
                writer.writerow(["t0", "t1", "t2"])
            writer.writerow([t0,t1,t2])



def log_csv(ifce_name: str, mode: int, test_id: int, chunk_gap: float, epochs_cur: int, epochs: int,
            n_split: int, n_combiner: int, used_chunk_num: int, used_data_len:int, used_data_shape, 
            cache_time: float, process_time: float, is_update:bool=False):
    file_name = "./measurements/testcase_" + \
        str(test_id) + "/proc_info_" + ifce_name + ".csv"

    if is_update:
        delLastLine(file_name)

    with open(file_name, "a+") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")

        if os.path.getsize(file_name) == 0:
            # leer: need header
            if index_dict_3[ifce_name] == 0:
                # client
                writer.writerow(["mode", "test id" , "chunk gap", "epochs current", "epochs", "n_split_client", "n_combiner_client",
                                "used chunk num", "used data length", "used data shape", "cache time", "processing time"])
            elif index_dict_3[ifce_name] == 4:
                # server
                writer.writerow(["mode", "test id", "chunk gap", "epochs current", "epochs", "n_split_server", "n_combiner_server",
                                "used chunk num", "used data length", "used data shape", "cache time", "processing time"])
            else:
                # vnf
                vnf_str = "_vnf" + str(index_dict_3[ifce_name] - 1)
                writer.writerow(["mode", "test id", "chunk gap", "epochs current", "epochs", "n_split" + vnf_str, "n_combiner" + vnf_str,
                                "used chunk num", "used data length", "used data shape", "cache time", "processing time"])
        writer.writerow([mode, test_id, chunk_gap, epochs_cur, epochs, n_split, n_combiner, used_chunk_num, used_data_len, used_data_shape, round(cache_time,3), round(process_time,3)])


def delLastLine(path):
    with open(path, "rb+") as f:
        lines = f.readlines()  # 读取所有行
        last_line = lines[-1]  # 取最后一行
        for i in range(len(last_line) + 2):  ##愚蠢办法，但是有效
            f.seek(-1, os.SEEK_END)
            f.truncate()
        f.close()
        f = open(path, "a")
        f.write('\r\n')
        f.close()

    return