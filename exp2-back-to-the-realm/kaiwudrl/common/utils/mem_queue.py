#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file mem_queue.py
# @brief
# @author kaiwu
# @date 2023-11-28


import random
import copy
import ctypes
from multiprocessing import Value, Array, Queue
import sys
import time
import numpy as np


class MemQueue(object):
    """
    FIFO get sample
    """

    def __init__(self, max_sample_num, sample_size, logger):
        self._maxlen = int(max_sample_num)
        self._sample_size = int(sample_size)
        self._data_queue = Queue(self._maxlen)
        self.logger = logger

    def __len__(self):
        return self._data_queue.qsize()

    def append(self, data):
        try:
            # self._data_queue.put(data, block=False)
            self._data_queue.put(data)
        except Exception:  # pylint: disable=broad-except
            error = sys.exc_info()[0]
            self.logger.exception("MemQueue append error {}".format(error))

    def get_sample(self):
        return self._data_queue.get()

    def clear(self):
        while not self._data_queue.empty():
            self._data_queue.get()

    def get_speed(self):
        return None
