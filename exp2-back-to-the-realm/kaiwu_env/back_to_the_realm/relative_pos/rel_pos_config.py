#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
@Project :back_to_the_realm
@File    :rel_pos_config.py
@Author  :kaiwu
@Date    :2023/1/9 17:23 

'''

import json 
map_name="map_1"
with open(f"environment/feature_process/relative_pos/{map_name}_dist.json", "r") as f:
    REL_POS = json.load(f)