#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file rainbow_wrapper.py
# @brief
# @author kaiwu
# @date 2023-11-28


import yaml
from kaiwudrl.common.utils.rainbow_utils import RainbowUtils
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine


class RainbowWrapper(object):
    """
    该类为七彩石的包装类, 主要包括从七彩石获取数据, 由于每个进程都需要获取, 故这里做成通用的, 各个进程调用即可, 减少重复代码
    """

    def __init__(self, logger):
        # 如果是不使用七彩石, 则提前返回
        if not int(CONFIG.use_rainbow):
            logger.info(f" CONFIG.use_rainbow is False, so return")
            return

        self.rainbow_utils = RainbowUtils()
        self.rainbow_utils.init(
            CONFIG.rainbow_url,
            CONFIG.rainbow_app_id,
            CONFIG.rainbow_user_id,
            CONFIG.rainbow_secret_key,
            CONFIG.rainbow_env_name,
            logger,
        )

        logger.info(f" RainbowUtils {self.rainbow_utils.identity}")

    def rainbow_activate_single_process(self, process_name, logger):
        """
        单独的某个进程处理七彩石逻辑
        """
        if not int(CONFIG.use_rainbow):
            logger.info(f" CONFIG.use_rainbow is False, so return")
            return

        result_code, data, result_msg = self.rainbow_utils.read_from_rainbow(process_name)
        if result_code:
            logger.error(
                f"read_from_rainbow failed, process_name is {process_name}, msg is {result_msg}, "
                f"result_code is {result_code}"
            )
            return

        if not data:
            logger.error(f"read_from_rainbow failed, process_name is {process_name}, data is None or data len is 0")
            return

        # 注意返回的格式成为list里带上map形式了
        for content in data:
            # 更新内存里的值, 再更新配置文件
            key_values_list = content.get("key_values", None)
            if key_values_list:
                for key_value in key_values_list:
                    value = key_value.get("value", None)
                    if value:
                        to_change_key_values = yaml.load(value, Loader=yaml.SafeLoader)
                        CONFIG.write_to_config(to_change_key_values)
                        CONFIG.save_to_file(process_name, to_change_key_values)

        logger.info(f" {process_name} CONFIG process_name is {process_name}, save_to_file success")
