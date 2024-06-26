#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   yohobuffer.py
@Time    :   2022/01/01 17:11:09
@Author  :   Jiakang Weng
@Version :   1.0
@Contact :   jiakang.weng@mailbox.tu-dresden.de
@License :   (C)Copyright 2021-2022
@Desc    :   bytes buffer to save bytes stream from last node, then use pickle load them all.
'''

# here put the import lib
import numpy as np
import threading
import copy


class YOHOBuffer():
    def __init__(self):
        self.length = 0
        self.buffer = bytes()
        self.lock = threading.Lock()

    def init(self):
        self.lock.acquire()
        # print(f"*** [log]: yohobuffer init !")
        self.length = 0
        self.buffer = bytes()
        self.lock.release()

    def clear_buffer(self):
        self.lock.acquire()
        # print(f"*** [log]: yohobuffer clear !")
        self.length = 0
        self.buffer = bytes()
        self.lock.release()

    def put(self, x:bytes):
        self.lock.acquire()
        # _size = self.length + x.shape[1]
        self.buffer += x
        self.length = len(self.buffer)
        self.lock.release()

    def extract(self):
        self.lock.acquire()
        # print(f"*** [log]: yohobuffer length = {len(self.buffer)}")
        out = copy.deepcopy(self.buffer)
        self.lock.release()
        # print(f"*** [log]: yohobuffer extract length {len(out)}")
        return out

    def extract_n(self, n):
        self.lock.acquire()
        out = copy.deepcopy(self.buffer[0:int(n)])
        self.lock.release()
        return out
    
    def atomic_put_last(self, x:bytes):
        self.lock.acquire()
        self.buffer += x
        self.length = len(self.buffer)
        out = copy.deepcopy(self.buffer)
        self.buffer = bytes()
        self.length = 0
        self.lock.release()
        return out

    def acquire(self):
        self.lock.acquire()

    def release(self):
        self.lock.release()

    def size(self):
        return self.length