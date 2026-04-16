#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file singleton.py
# @brief
# @author kaiwu
# @date 2023-11-28


# 单例模式


import threading


class Singleton:
    _lock = threading.Lock()
    _instances = {}  # 存储所有被装饰类的单例实例

    def __init__(self, cls):
        self.cls = cls

    def __call__(self, *args, **kwargs):
        """支持带参数的构造函数"""
        # 双检锁确保线程安全
        if self.cls not in Singleton._instances:
            with Singleton._lock:
                if self.cls not in Singleton._instances:
                    Singleton._instances[self.cls] = self.cls(*args, **kwargs)
        return Singleton._instances[self.cls]

    def __instancecheck__(self, instance):
        """支持isinstance检查"""
        return isinstance(instance, self.cls)
