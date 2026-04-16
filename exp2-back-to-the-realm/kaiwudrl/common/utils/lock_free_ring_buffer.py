#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file lock_free_ring_buffer.py
# @brief
# @author kaiwu
# @date 2023-11-28


import torch
import ctypes
import numpy as np
from ctypes import c_uint64, Structure
from ctypes.util import find_library
from typing import Optional, List, Any
import threading

libc = ctypes.CDLL(find_library("c"))


class AtomicCounter(Structure):
    _fields_ = [("counter", c_uint64)]

    def cas(self, old, new):
        return libc.__sync_bool_compare_and_swap(ctypes.byref(self.counter), old, new)


class TorchLockFreeRingBuffer:
    """支持PyTorch张量的无锁环形缓冲区"""

    def __init__(self, capacity, sample_shape, device="cuda"):
        self.capacity = capacity
        self.device = torch.device(device)
        pin_memory = not torch.cuda.is_available()
        self.storage = torch.empty(
            (capacity, *sample_shape), dtype=torch.float32, device=self.device, pin_memory=pin_memory
        )
        self.head = AtomicCounter()
        self.tail = AtomicCounter()

        # 缓存行填充
        self._pad = torch.zeros(64 - ctypes.sizeof(AtomicCounter), dtype=torch.uint8)

    def put_batch(self, batch: torch.Tensor) -> bool:
        batch_size = batch.size(0)
        while True:
            current_head = self.head.counter
            current_tail = self.tail.counter
            available = self.capacity - (current_head - current_tail)

            if available < batch_size:
                return False

            new_head = current_head + batch_size
            if self.head.cas(current_head, new_head):
                # 分段写入优化
                wrap_pos = new_head % self.capacity
                if wrap_pos < batch_size:
                    self.storage[current_head:] = batch[: self.capacity - current_head]
                    self.storage[:wrap_pos] = batch[self.capacity - current_head :]
                else:
                    self.storage[current_head:new_head] = batch
                return True

    def get_batch(self, batch_size: int) -> torch.Tensor:
        while True:
            current_tail = self.tail.counter
            current_head = self.head.counter
            available = current_head - current_tail

            if available == 0:
                return torch.empty(0, device=self.device)

            read_size = min(available, batch_size)
            if self.tail.cas(current_tail, current_tail + read_size):
                # 零拷贝视图
                wrap_pos = (current_tail + read_size) % self.capacity
                if wrap_pos < read_size:
                    return torch.cat([self.storage[current_tail:], self.storage[:wrap_pos]])
                else:
                    return self.storage[current_tail : current_tail + read_size]


class ThreadSafeRingBuffer:
    """
    线程安全环形缓冲区
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0  # 写入指针
        self.tail = 0  # 读取指针
        self.lock = threading.Lock()

    @property
    def free_space(self):
        """当前剩余可用空间（线程安全计算）"""
        with self.lock:
            # 计算已用空间
            used = (self.head - self.tail) % self.capacity
            # 剩余空间 = 总容量 - 已用空间
            return self.capacity - used

    def put_batch(self, batch):
        with self.lock:
            batch_size = len(batch)
            # 覆盖写入逻辑
            for i in range(batch_size):
                self.buffer[(self.head + i) % self.capacity] = batch[i]
            self.head = (self.head + batch_size) % self.capacity

    def get_batch(self, batch_size):
        with self.lock:
            available = (self.head - self.tail) % self.capacity
            if available < batch_size:
                return None
            indices = [(self.tail + i) % self.capacity for i in range(batch_size)]
            batch = [self.buffer[i] for i in indices]
            self.tail = (self.tail + batch_size) % self.capacity
            return batch


class ThreadSafeRingBufferV2:
    """线程安全环形缓冲区（兼容CPU/GPU）"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0  # 写入位置
        self.tail = 0  # 读取位置
        self.lock = threading.Lock()
        self.available = threading.Condition(self.lock)

    def put_batch(self, data: List[Any]) -> int:
        """批量写入数据，返回成功写入数量"""
        with self.lock:
            space = self.capacity - (self.head - self.tail)
            insert_size = min(len(data), space)

            if insert_size == 0:
                return 0

            # 计算写入位置
            pos = self.head % self.capacity
            end_pos = pos + insert_size

            if end_pos <= self.capacity:
                self.buffer[pos:end_pos] = data[:insert_size]
            else:
                # 环形回绕处理
                part1_size = self.capacity - pos
                self.buffer[pos:] = data[:part1_size]
                self.buffer[: end_pos - self.capacity] = data[part1_size:insert_size]

            self.head += insert_size
            self.available.notify_all()
            return insert_size

    def get_batch(self, size: int, timeout: Optional[float] = None) -> List[Any]:
        """批量读取数据，支持超时等待"""
        with self.available:
            # 等待足够数据
            while (self.head - self.tail) < size:
                if not self.available.wait(timeout=timeout):
                    return []

            # 计算读取位置
            pos = self.tail % self.capacity
            end_pos = pos + size

            if end_pos <= self.capacity:
                result = self.buffer[pos:end_pos]
            else:
                # 处理环形回绕
                part1 = self.buffer[pos:]
                part2 = self.buffer[: end_pos - self.capacity]
                result = part1 + part2

            self.tail += size
            return result


class HighPerfRingBufferV1:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = [None] * self.capacity
        self.head = 0  # 无锁原子计数器
        self.tail = 0  # 无锁原子计数器
        self.write_lock = threading.Lock()
        self.read_lock = threading.Lock()

    def put_batch(self, data):
        with self.write_lock:  # 分段锁
            # 仅保护head修改
            new_head = self.head + len(data)
            # 写入数据（无锁）
            for i in range(len(data)):
                self.buffer[(self.head + i) % self.capacity] = data[i]
            self.head = new_head  # 原子操作

    def get_batch(self, size):
        with self.read_lock:
            avail = self.head - self.tail
            if avail < size:
                return []
            # 读取数据（无锁）
            result = [self.buffer[(self.tail + i) % self.capacity] for i in range(size)]
            self.tail += size  # 原子操作
            return result


class HighPerfRingBufferV2:
    def __init__(self, capacity: int, sample_shape: tuple, dtype=np.float32):
        self.capacity = capacity
        self.buffer = np.zeros((capacity, *sample_shape), dtype=dtype)
        self.head = 0  # 绝对写入位置（永不取模）
        self.tail = 0  # 绝对读取位置（永不取模）
        self.lock = threading.Lock()

    @property
    def size(self) -> int:
        """有效数据量（确保非负）"""
        return max(0, self.head - self.tail)

    def put_batch(self, batch: np.ndarray) -> None:
        """线程安全写入批次数据（自动覆盖旧数据）"""
        with self.lock:
            batch_size = batch.shape[0]
            if batch_size == 0:
                return

            # 计算需要覆盖的旧数据量
            free_space = self.capacity - self.size
            if batch_size > free_space:
                overwrite = batch_size - free_space
                self.tail += overwrite  # 直接移动绝对位置

            # 计算环形写入位置
            start_idx = self.head % self.capacity
            end_idx = (self.head + batch_size) % self.capacity

            # 写入数据（处理环形回绕）
            if start_idx + batch_size <= self.capacity:
                self.buffer[start_idx : start_idx + batch_size] = batch
            else:
                split = self.capacity - start_idx
                self.buffer[start_idx:] = batch[:split]
                self.buffer[:end_idx] = batch[split:]

            # 更新绝对头指针
            self.head += batch_size

    def get_contiguous_batch(self, batch_size: int) -> Optional[np.ndarray]:
        """线程安全读取连续批次数据（返回None表示数据不足）"""
        with self.lock:
            if self.size < batch_size:
                return None

            # 计算环形读取位置
            start_idx = self.tail % self.capacity
            end_idx = (self.tail + batch_size) % self.capacity

            # 读取数据（处理环形回绕）
            if start_idx + batch_size <= self.capacity:
                batch = self.buffer[start_idx : start_idx + batch_size].copy()
            else:
                part1 = self.buffer[start_idx:]
                part2 = self.buffer[:end_idx]
                batch = np.concatenate([part1, part2], axis=0)

            # 更新绝对尾指针
            self.tail += batch_size
            return batch

    def clear(self) -> None:
        """清空缓冲区"""
        with self.lock:
            self.head = 0
            self.tail = 0
