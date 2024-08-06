#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   packetutils.py
@Time    :   2022/01/04 11:55:33
@Author  :   Jiakang Weng
@Version :   1.0
@Contact :   jiakang.weng@mailbox.tu-dresden.de
@License :   (C)Copyright 2021-2022
@Desc    :   None
'''

# here put the import lib


from curses import def_prog_mode
import numpy as np
import pickle
import math
import torch
import struct
import logging
#from sklearn import preprocessing
#import librosa
from utils.yoholog import *
import pickle
import os
from PIL import Image
import torch
from torchvision import transforms

# HEADER DEF, 9 bits, but no.9 bit we don't use
HEADER_SPLIT_DATA =         0b00000001               # split data chunk [DATA]
HEADER_SPLIT_FINISH =       0b00000010               # split data chunk [FINISH]
HEADER_COMBINER_DATA =      0b00000100               # combiner data chunk [DATA]
HEADER_COMBINER_FINISH =    0b00001000               # combiner data chunk [FINISH]
HEADER_INIT =               0b00010000               # first chunk: meta data(init_setting)
HEADER_DATA =               0b00100000               # data chunk: X matrix 
HEADER_FINISH =             0b01000000               # last chunk
HEADER_CLEAR_CACHE =        0b10000000               # clear cache command


MTU = 1024  # 1400
LENGTH = 65536 #64000
BATCH_SIZE = 1
HEAD_GAP = 0.002  # 0.05
CHUNK_GAP = 0.001  # 0.01
source_dict = {0: "pump", 1: "slider", 2: "fan", 3: "valve"}
IA_NET = False

is_debug = False

def load_audio(path, length=LENGTH):
    audio, _ = librosa.load(path, sr=16000, mono=True)
    target_length = audio.shape[0]
    start_pos = np.random.randint(0, max(target_length - length + 1, 1))
    end_pos = start_pos + length
    return audio[start_pos:end_pos]


def get_chunks(data_in):
    pkg = pickle.dumps(data_in)
    num_chunks = math.ceil(len(pkg) / MTU)
    return num_chunks, pkg



def collect_data_fixed(source_id, name, id_):
    if name not in ["normals", "abnormals"]:
        raise ValueError("The type must be normals or abnormals!")
    if not 0 <= id_ <= 5:
        raise ValueError("The id should between 0 and 5!")
    path = "./demo_files/{}/{}/{}.wav"
    path = path.format(name, source_dict[source_id], id_)
    data = load_audio(path, length=LENGTH)
    return data


def get_network_data_tensors(batch_size=BATCH_SIZE):
    scaler = preprocessing.MaxAbsScaler()
    mix = np.zeros(LENGTH, dtype=np.float32)

    if is_debug: print("Collect source data from ../demo_files.")
    for i in range(4):
        data = collect_data_fixed(i, "normals", 0)
        mix += data
    mix = scaler.fit_transform(mix.reshape(-1, 1)).T

    mix = torch.tensor(mix).unsqueeze(0)
    if batch_size > 0:
        mix = mix.repeat(batch_size, 1, 1)
    return mix

def get_mlp_data_tensors():
    # data = torch.randint(0, 255, [64, 3072], dtype=torch.float32) / 255.0
    
    with open("./demo_files/data.pkl", 'rb') as f:
        data_dict = pickle.load(f)
    data = data_dict["data"]
    data = data[:64, ...].reshape(-1, 3072)
    
    return data

def get_yolo_data_tensors():
    image_dir = './dataset/fruits/valid/images'
    image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(('jpg', 'jpeg', 'png'))]

    file_data_list = []
    for index, file_path in enumerate(image_files):
        with open(file_path, 'rb') as file:
            if index == len(image_files) - 1:

                serialized_data = HEADER_COMBINER_FINISH.to_bytes(1, byteorder='big') + pickle.dumps(file.read())
            else: 
                serialized_data = HEADER_COMBINER_DATA.to_bytes(1, byteorder='big') + pickle.dumps(file.read())
            
            file_data_list.append(serialized_data)

        
    return file_data_list



class ChunkHandler():

    def __init__(self) -> None:
        pass

    def float2bytes(self, f):
        bs = struct.pack("f",f)
        return (bs[3],bs[2],bs[1],bs[0])

    def floatlist2bytes(self, list_f):
        ret = bytearray()
        for i in range(len(list_f)):
            bs3, bs2, bs1, bs0 = self.float2bytes(list_f[i])
            ret.append(bs3)
            ret.append(bs2)
            ret.append(bs1)
            ret.append(bs0)
        return ret

    def bytes2float(self,h1,h2,h3,h4):
        ba = bytearray()
        ba.append(h1)
        ba.append(h2)
        ba.append(h3)
        ba.append(h4)
        return struct.unpack("!f",ba)[0]

    def bytes2floatlist(self, bytes_in):
        float_list = []
        float_bytes_list = [bytes_in[i*4:(i+1)*4] for i in range( math.ceil(len(bytes_in)/4) )]
        for i in range(len(float_bytes_list)):
            bs3 = float_bytes_list[i][0]
            bs2 = float_bytes_list[i][1]
            bs1 = float_bytes_list[i][2]
            bs0 = float_bytes_list[i][3]
            float_list.append(self.bytes2float(bs3, bs2, bs1, bs0))
        return float_list
      
    def get_serialize_imagefile(self, header, image_list):
        chunk_list = [] 
        for i in range(len(image_list)):
            bytes_data_list = self._get_substream_arr(image_list[i], MTU)
            for j in range(len(bytes_data_list)):
                chunk_list.append(self.initial_chunk(header, bytes_data_list[j]))

        # Support HEADER_FINISH in last
        if header == HEADER_DATA:
            chunk_list[-1] = bytes([HEADER_FINISH]) + chunk_list[-1][1:]
        elif header == HEADER_COMBINER_DATA:
            chunk_list[-1] = bytes([HEADER_COMBINER_FINISH]) + chunk_list[-1][1:]
        return chunk_list
    
    def get_serialize_torcharray(self, header, index, data):
        if len(data.size()) == 2:
            data = data.unsqueeze(0)
        elif len(data.size()) == 3:
            data = data.unsqueeze(0)
            data = data.unsqueeze(0)
        data_sq = data.reshape(1,-1).squeeze()
        data_np = data_sq.detach().numpy().tolist()
        if type(data_np) != type([0]): data_np = [data_np]
        # log_file("packetutils").debug(f'type(data_np) = {type(data_np)}')
        bytes_data =  self.floatlist2bytes(data_np)  
        bytes_data_list = self._get_substream_arr(bytes_data, MTU)
        chunk_list = [] 
        for i in range(len(bytes_data_list)):
            chunk_list.append(self.initial_chunk(header, bytes_data_list[i], index)) 
        # Support HEADER_FINISH in last
        if header == HEADER_DATA:
            chunk_list[-1] = bytes([HEADER_FINISH]) + chunk_list[-1][1:]
        elif header == HEADER_COMBINER_DATA:
            chunk_list[-1] = bytes([HEADER_COMBINER_FINISH]) + chunk_list[-1][1:]
        return chunk_list
    
    def initial_chunk(self, header, chunk=None, index=None):
        '''
            return the bytes with [header, chunk]
            _in_:
                header: HEADER_INIT, HEADER_DATA, HEADER_FINISH, check define for more info
                chunk: chunk shoulde be use pickle.dumps(chunk) before.
            _out_:
                bytes: HEADER + [index] + chunk
        '''
        if chunk == None:
            return bytes([header])

        if len(chunk) > MTU:
            raise ValueError(f'package size={len(chunk)} is bigger than MTU={MTU} !')

        if index == None:
            return bytes([header]) + chunk

        return bytes([header]) + bytes([index]) + chunk

    def _get_substream_arr(self, X, m_substream):
        substream_num = math.ceil(len(X)/m_substream)
        return [X[i*m_substream:(i+1)*m_substream] for i in range(substream_num)]

    def get_chunks_metadata(self, init_settings):
        
        return self.initial_chunk(HEADER_INIT, pickle.dumps(init_settings))

    def get_chunks_fc(self, data):
        chunk_arr = self._get_substream_arr(pickle.dumps(data), MTU)
        ret = []
        for chunk in chunk_arr:
            ret.append(self.initial_chunk(HEADER_DATA, chunk, None))
        ret[-1] = bytes([HEADER_FINISH]) + ret[-1][1:]
        return ret

    def get_chunks_clean(self):
        return self.initial_chunk(HEADER_CLEAR_CACHE)

    def derialize_with_index(self, bytes_in):
    
        data = self.bytes2floatlist(bytes_in)
        data = torch.tensor(data)
        data = data.unsqueeze(0)
        data = data.unsqueeze(0)
        return data

class YoloChunkHandler():

    def __init__(self) -> None:
        pass

    def float2bytes(self, f):
        bs = struct.pack("f", f)
        return (bs[3], bs[2], bs[1], bs[0])

    def floatlist2bytes(self, list_f):
        ret = bytearray()
        for i in range(len(list_f)):
            bs3, bs2, bs1, bs0 = self.float2bytes(list_f[i])
            ret.append(bs3)
            ret.append(bs2)
            ret.append(bs1)
            ret.append(bs0)
        return ret

    def bytes2float(self, h1, h2, h3, h4):
        ba = bytearray()
        ba.append(h1)
        ba.append(h2)
        ba.append(h3)
        ba.append(h4)
        return struct.unpack("!f", ba)[0]

    def bytes2floatlist(self, bytes_in):
        float_list = []
        float_bytes_list = [bytes_in[i*4:(i+1)*4] for i in range(math.ceil(len(bytes_in)/4))]
        for i in range(len(float_bytes_list)):
            bs3 = float_bytes_list[i][0]
            bs2 = float_bytes_list[i][1]
            bs1 = float_bytes_list[i][2]
            bs0 = float_bytes_list[i][3]
            float_list.append(self.bytes2float(bs3, bs2, bs1, bs0))
        return float_list
      
    def get_serialize_torcharray(self, header, index, data_list):
        chunk_list = []

        for i, image_tensor in enumerate(data_list):
            if image_tensor.size() != torch.Size([3, 416, 416]):
                raise ValueError("Each tensor must have shape [3, 416, 416]")

            data_sq = image_tensor.reshape(1, -1).squeeze()
            data_np = data_sq.detach().numpy().tolist()
            if type(data_np) != type([0]): 
                data_np = [data_np]
            bytes_data = self.floatlist2bytes(data_np)
            bytes_data_list = self._get_substream_arr(bytes_data, MTU)

            for j in range(len(bytes_data_list)):
                chunk_list.append(self.initial_chunk(header, bytes_data_list[j], index + i))

            
        chunk_list[-1] = bytes([HEADER_COMBINER_FINISH]) + chunk_list[-1][1:]
        return chunk_list
    
    def initial_chunk(self, header, chunk=None, index=None):
        if chunk is None:
            return bytes([header])

        if len(chunk) > MTU:
            raise ValueError(f'package size={len(chunk)} is bigger than MTU={MTU} !')

        if index is None:
            return bytes([header]) + chunk

        return bytes([header]) + index.to_bytes(4, byteorder='big') + chunk

    def _get_substream_arr(self, X, m_substream):
        substream_num = math.ceil(len(X) / m_substream)
        return [X[i*m_substream:(i+1)*m_substream] for i in range(substream_num)]

    def get_chunks_metadata(self, init_settings):
        return self.initial_chunk(HEADER_INIT, pickle.dumps(init_settings))

    def get_chunks_fc(self, data):
        chunk_arr = self._get_substream_arr(pickle.dumps(data), MTU)
        ret = []
        for chunk in chunk_arr:
            ret.append(self.initial_chunk(HEADER_DATA, chunk, None))
        ret[-1] = bytes([HEADER_FINISH]) + ret[-1][1:]
        return ret

    def get_chunks_clean(self):
        return self.initial_chunk(HEADER_CLEAR_CACHE)

    def derialize_with_index(self, bytes_in):
        data = self.bytes2floatlist(bytes_in)
        data = torch.tensor(data)
        data = data.view(3, 416, 416)
        return data

class YoloTxtChunkHandler:

    def __init__(self):
        self.MTU = MTU

    def str_to_bytes(self, data_str):
        """Convert string data to bytes."""
        return data_str.encode('utf-8')

    def bytes_to_str(self, data_bytes):
        """Convert bytes data to string."""
        return data_bytes.decode('utf-8')

    def chunk_data(self, data_bytes):
        """Split data bytes into chunks based on MTU."""
        return [data_bytes[i:i + self.MTU] for i in range(0, len(data_bytes), self.MTU)]

    def serialize_list(self, data_list):
        """Serialize list of strings into chunks."""
        serialized_chunks = []
        for data_str in data_list:
            data_bytes = self.str_to_bytes(data_str)
            chunks = self.chunk_data(data_bytes)
            serialized_chunks.extend(chunks)
        return serialized_chunks

    def deserialize_chunks(self, chunks):
        """Deserialize chunks back into list of strings."""
        data_bytes = b''.join(chunks)
        data_str = self.bytes_to_str(data_bytes)
        return data_str.split('\n') if data_str else []
    

chunk_handler = ChunkHandler()
yolo_chunk_handle = YoloChunkHandler()
yolo_txt_chunk_handel = YoloTxtChunkHandler()
