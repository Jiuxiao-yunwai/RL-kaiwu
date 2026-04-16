#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import toml
from agent_target_dqn.conf.conf import Config


def read_usr_conf(usr_conf_file, logger):
    """
    read usr conf
    读取配置文件
    """
    if not usr_conf_file or not os.path.exists(usr_conf_file):
        return None

    try:
        with open(usr_conf_file, "r") as file:
            loaded_usr_conf = toml.load(file)
    except Exception as e:
        logger.exception(f"toml.load failed, error msg is {str(e)}, {usr_conf_file} please check")
        return None

    try:
        usr_conf = {"env_conf": {}}
        for key, value in loaded_usr_conf["env_conf"].items():

            if key == "treasure_id":
                if type(value) is not list:
                    logger.error(f"{key}: {value} is not a list, please check")
                    return None

                if any(type(point) is not int for point in value):
                    logger.error("Each element in treasure_id must be an integer, please check")
                    return False

                # 宝箱ID需要单独处理每个元素里面的数字
                new_value = []
                for v in value:
                    if isinstance(v, float):
                        if abs(v - round(v)) < 1e-20:
                            v = int(round(v))
                        else:
                            logger.error(f"{key}: {value} {v} is a float not close to an integer, please check")
                            return None
                    new_value.append(v)
                value = new_value
                usr_conf["env_conf"][key] = value
            elif key in ["treasure_random", "hidden_status"]:
                if isinstance(value, bool):
                    usr_conf["env_conf"][key] = value
                else:
                    logger.error(f"{key}: {value} is not a boolean, please check")
                    return None
            else:
                if type(value) is not int:
                    logger.error(
                        "Each element in start/end/max_step/treasure_count/talent_type must be an integer, please check"
                    )
                    return False
                if isinstance(value, float):
                    if abs(value - round(value)) < 1e-20:
                        value = int(round(value))
                    else:
                        logger.error(f"{key}: {value} is a float not close to an integer, please check")
                        return None
                if type(value) is not int:
                    logger.error(f"{key}: {value} is not int, please check")
                    return None
                usr_conf["env_conf"][key] = int(value)

        usr_conf["env_conf"]["view_size"] = Config.VIEW_SIZE

        return usr_conf
    except Exception as e:
        logger.exception(f"read_usr_conf failed, {usr_conf_file}, {str(e)}, please check")
        return None


def check_usr_conf(usr_conf, logger):
    if not usr_conf:
        logger.error("usr_conf is None, please check")
        return False

    env_conf = usr_conf.get("env_conf", {})
    start = env_conf.get("start")
    end = env_conf.get("end")
    treasure_id = env_conf.get("treasure_id")
    talent_type = env_conf.get("talent_type")
    treasure_count = env_conf.get("treasure_count")
    treasure_random = env_conf.get("treasure_random")
    max_step = env_conf.get("max_step")

    if (
        treasure_random is None
        or treasure_count is None
        or treasure_id is None
        or start is None
        or end is None
        or max_step is None
        or talent_type is None
    ):
        logger.error(
            f"treasure_random or treasure_count or treasure_id or start or end or max_step or talent_type is None, please check"
        )
        return False

    if any(type(id) is not int for id in treasure_id) or len(treasure_id) != len(set(treasure_id)):
        logger.error("Each element in treasure_id must be an integer and there must be no duplicates, please check")
        return False

    if treasure_id and not set(treasure_id).issubset(set(range(1, 16))):
        logger.error("Elements in treasure_id should be between 1 and 15, please check")
        return False

    if type(treasure_count) is not int or not (0 <= treasure_count <= 13):
        logger.error("treasure_count should be between 0 and 13, please check")
        return False

    if not isinstance(treasure_random, bool):
        logger.error("treasure_random field can only be false or true, please check")
        return False

    if start is not None and end is not None and start == end:
        logger.error("Start and end points should not be the same, please check")
        return False

    if type(start) is not int or (start < 1 or start > 15):
        logger.error("Start should be an intergar and between 1 and 15, please check")
        return False

    if type(end) is not int or (end < 1 or end > 15):
        logger.error("End should be an intergar and between 1 and 15, please check")
        return False

    if type(talent_type) is not int or talent_type != 1:
        logger.error("talent_type field can only be intergar 1, please check")
        return False

    if max_step is not None:
        if type(max_step) is not int:
            logger.error("max_step should be an integer, please check")
            return False

        try:
            max_step = int(max_step)
            if max_step < 1:
                logger.error("max_step should not be negative or 0, please check")
                return False

            if max_step > 2000:
                logger.error("max_step should not be larger than 2000, please check")
                return False
        except ValueError:
            logger.error("max_step should be an integer, please check")
            return False

    if not treasure_random:
        if not set(treasure_id).issubset(set([i for i in range(1, 16)])):
            logger.error("treasure_random field can only be 0 or 1, please check")
            return False

        if start in treasure_id or end in treasure_id:
            logger.error(
                f"treasure_id should not include the start or end points, set to {treasure_id}, start {start}, end {end}"
            )
            return False

    if treasure_random:
        if treasure_count not in [i for i in range(0, 14)]:
            logger.error(f"treasure_count should be between 0 and 13, set to {treasure_count}")
            return False

    return True
