#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file reverb_dataset_v2.py
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
    目前为止, 性能稳定版本, 实地测试V100上单次训练耗时200ms左右
    该版本的优化情况:
    关键优化点:
    1. 双缓冲队列设计
    2. 动态批量处理
    3. 异步GPU流水线

    不足:
    1. 缓冲区设计缺陷
    2. 异常恢复机制缺失
    3. torch.stack耗时比较多
    """

    def __init__(self, logger):
        super().__init__()
        self._table_names = ["{}_{}".format(CONFIG.reverb_table_name, i) for i in range(int(CONFIG.reverb_table_size))]

        self.is_gpu_machine = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.is_gpu_machine else "cpu")
        self.logger = logger

        # 双缓冲队列（线程安全）
        self.buffer_lock = threading.Lock()
        # 主缓冲
        self.active_buffer = deque(maxlen=CONFIG.train_batch_size * 4)
        # 后台缓冲
        self.backup_buffer = deque(maxlen=CONFIG.train_batch_size * 4)

        # 线程停止信号
        self._stop_event = threading.Event()

        self.client = None

        # 后台填充线程
        self._fill_thread = None

        # GPU流
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def start_background_filler(self):
        """启动后台填充线程"""
        if self._fill_thread is None:
            self._stop_event.clear()

            self.client = reverb.Client(f"localhost:{CONFIG.reverb_svr_port}")
            self._fill_thread = threading.Thread(target=self._fill_buffers_loop, daemon=True)
            self._fill_thread.start()
            self.logger.info(
                f"start_background_filler success, reverb.Client connect at localhost:{CONFIG.reverb_svr_port}"
            )

    def _fill_buffers_loop(self):
        """后台线程持续填充缓冲区"""

        while not self._stop_event.is_set():
            # 填充备用缓冲区
            self._fill_buffer(self.backup_buffer)

            # 交换缓冲区（原子操作）
            with self.buffer_lock:
                self.active_buffer, self.backup_buffer = self.backup_buffer, self.active_buffer

            # 避免高频切换
            time.sleep(CONFIG.idle_sleep_second)

    def _fill_buffer(self, target_buffer):
        """填充指定缓冲区至目标大小"""
        while len(target_buffer) < target_buffer.maxlen and not self._stop_event.is_set():
            try:
                # 动态设置批量处理大小
                data = self.client.sample(
                    table=self._table_names[0],
                    num_samples=min(2 * CONFIG.train_batch_size, target_buffer.maxlen - len(target_buffer)),
                )
                if data:
                    # 异步转换为 PyTorch 张量
                    for sample in data:
                        target_buffer.append(torch.from_numpy(sample[0].data[0]).float())
                else:
                    time.sleep(CONFIG.idle_sleep_second)
            except Exception as e:
                self.logger.error(f"后台填充线程错误: {str(e)}")
                time.sleep(CONFIG.idle_sleep_second)

    def __iter__(self):

        # 启动后台线程从reverb读取数据
        self.start_background_filler()

        while True:
            # 从主缓冲取数据
            with self.buffer_lock:
                if len(self.active_buffer) >= CONFIG.train_batch_size:
                    # 批量获取
                    batch = [self.active_buffer.popleft() for _ in range(CONFIG.train_batch_size)]
                else:
                    batch = None

            if batch is None:
                time.sleep(CONFIG.idle_sleep_second)
                continue

            if not self.is_gpu_machine:
                # CPU路径
                tensor_batch = torch.stack(batch, dim=0).to(dtype=torch.float32)
            else:
                # GPU路径
                with torch.cuda.stream(self.stream):
                    # 启用 pin_memory
                    tensor_batch = torch.stack(batch, dim=0).to(dtype=torch.float32).pin_memory()
                    tensor_batch = tensor_batch.to(device=self.device, non_blocking=True)

            yield [tensor_batch]

    def __del__(self):
        self._stop_event.set()
        if self._fill_thread is not None:
            self._fill_thread.join(timeout=5)

    def get_metrics(self):
        """获取性能指标"""
        return {"buffer_utilization": f"{len(self.active_buffer)}/{self.active_buffer.maxlen}"}
