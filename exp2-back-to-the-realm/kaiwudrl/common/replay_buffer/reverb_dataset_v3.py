#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file reverb_dataset_v3.py
# @brief
# @author kaiwu
# @date 2023-11-28


from collections import deque
import torch
import numpy as np
import reverb
import threading
import time
from typing import Optional, List, Any
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.lock_free_ring_buffer import HighPerfRingBufferV2


class ReverbDataset(torch.utils.data.IterableDataset):
    """
    目前为止, 性能最优版本, 实地测试V100上单次训练耗时60ms左右
    性能优化关键点:
    1. 双缓冲并行处理
    2. GPU异步流水线
    3. 批量数据获取

    不足:
    1. 异常处理机制不足
    """

    def __init__(self, logger):
        super().__init__()
        self._table_names = ["{}_{}".format(CONFIG.reverb_table_name, i) for i in range(int(CONFIG.reverb_table_size))]
        self.is_gpu_machine = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.is_gpu_machine else "cpu")
        self.logger = logger

        # 连续内存双缓冲
        self.block_size = 4 * CONFIG.train_batch_size
        sample_shape = (CONFIG.sample_dim,)
        self.buffers = [
            torch.empty((self.block_size, *sample_shape), dtype=torch.float32, pin_memory=self.is_gpu_machine)
            for _ in range(2)
        ]
        self.write_idx = 0
        self.read_idx = 0
        # 每个缓冲区的有效数据长度
        self.write_cursors = [0, 0]
        # 缓冲区就绪状态
        self.ready_flags = [False, False]
        self.buffer_lock = threading.Lock()
        self.data_cond = threading.Condition()
        # 增加原子操作计数器
        self._atomic_version = [0, 0]

        # 线程控制, 只用单线程即可
        self._stop_event = threading.Event()
        self._fill_threads = None
        self._fill_threads_num = 1

        # GPU配置
        if self.is_gpu_machine:
            self.gpu_batch = torch.empty((CONFIG.train_batch_size, *sample_shape), device=self.device)
            self.stream = torch.cuda.Stream()
            self._cuda_events = [torch.cuda.Event() for _ in range(2)]

    def start_background_filler(self):
        if not self._fill_threads:
            self._stop_event.clear()
            self._fill_threads = [
                threading.Thread(
                    target=self._fill_buffers_loop,
                    args=(reverb.Client(f"localhost:{CONFIG.reverb_svr_port}"),),
                    daemon=True,
                )
                for _ in range(self._fill_threads_num)
            ]
            for t in self._fill_threads:
                t.start()

    def _fill_buffers_loop(self, client):
        while not self._stop_event.is_set():
            with self.buffer_lock:
                buffer_idx = self.write_idx
                current_buffer = self.buffers[buffer_idx]

                # 获取数据
                data = list(client.sample(table=self._table_names[0], num_samples=self.block_size))
                if not data:
                    time.sleep(CONFIG.idle_sleep_second)
                    continue

                valid_samples = min(len(data), self.block_size)

                # 写入内存
                numpy_view = current_buffer.numpy()
                numpy_view[:valid_samples] = [x[0].data[0] for x in data[:valid_samples]]

                # 原子更新版本号
                self._atomic_version[buffer_idx] += 1
                # 内存屏障确保写入可见性
                if self.is_gpu_machine and valid_samples > 0:
                    torch.cuda.current_stream().synchronize()

                # 更新状态
                self.write_cursors[buffer_idx] = valid_samples
                self.ready_flags[buffer_idx] = True
                self.write_idx = 1 - buffer_idx

            # 通知消费者线程
            with self.data_cond:
                self.data_cond.notify_all()

    def __iter__(self):
        self.start_background_filler()

        read_cursor = 0
        current_buffer = 0
        expected_version = 0

        while True:
            with self.data_cond:
                # 等待缓冲区就绪
                while not (
                    self.ready_flags[current_buffer] and self._atomic_version[current_buffer] > expected_version
                ):
                    self.data_cond.wait(timeout=CONFIG.idle_sleep_second)
                    if self._stop_event.is_set():
                        return
                expected_version = self._atomic_version[current_buffer]

            # 计算可用数据
            available = self.write_cursors[current_buffer] - read_cursor
            if available >= CONFIG.train_batch_size:

                # 直接内存访问
                batch = self.buffers[current_buffer][read_cursor : read_cursor + CONFIG.train_batch_size]
                read_cursor += CONFIG.train_batch_size

                # GPU传输优化
                if self.is_gpu_machine:
                    event = self._cuda_events[current_buffer]
                    with torch.cuda.stream(self.stream):
                        self.gpu_batch.copy_(batch, non_blocking=True)
                        event.record()

                    event.wait()
                    yield [self.gpu_batch]
                else:
                    yield [batch]

                # 缓冲区维护
                remaining = self.write_cursors[current_buffer] - read_cursor
                if remaining <= CONFIG.train_batch_size * 0.2:
                    with self.buffer_lock:
                        self.ready_flags[current_buffer] = False
                        current_buffer = 1 - current_buffer
                        read_cursor = 0
            else:
                # 快速切换逻辑
                alt_buffer = 1 - current_buffer
                if self.ready_flags[alt_buffer]:
                    current_buffer = alt_buffer
                    read_cursor = 0
                else:
                    time.sleep(0)

    def shutdown(self):
        self._stop_event.set()
        if self.is_gpu_machine:
            del self.gpu_batch
            torch.cuda.empty_cache()

    def __del__(self):
        self.shutdown()
        if self._fill_threads:
            for t in self._fill_threads:
                t.join(timeout=5)

    def get_metrics(self):
        """获取性能指标"""
        return {
            "buffer_utilization": sum(self.write_cursors) / (2 * self.block_size),
        }
