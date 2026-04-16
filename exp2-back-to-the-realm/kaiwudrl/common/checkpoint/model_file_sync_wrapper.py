#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file model_file_sync_wrapper.py
# @brief
# @author kaiwu
# @date 2023-11-28


from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.checkpoint.model_file_sync import ModelFileSync
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine


class ModelFileSyncWrapper(object):
    def __init__(self) -> None:

        self.ckpt_sync_warper = None

    def init(self):
        # 如果配置项中启用了模型池（modelpool），则实例化ModelFileSync对象并启动同步
        if CONFIG.ckpt_sync_way == KaiwuDRLDefine.CKPT_SYNC_WAY_MODELPOOL:
            self.ckpt_sync_warper = ModelFileSync()

            # 只有off-policy的场景下会启动ModelFileSync进程
            if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_OFF_POLICY:
                self.ckpt_sync_warper.start()
        # 如果未启用模型池，则将ckpt_sync_warper设置为NotImplemented
        else:
            self.ckpt_sync_warper = NotImplemented

    def stop(self):
        # 如果配置项中启用了模型池（modelpool），则停止模型文件同步
        if CONFIG.ckpt_sync_way == KaiwuDRLDefine.CKPT_SYNC_WAY_MODELPOOL:
            # 只有off-policy的场景下会启动ModelFileSync进程
            if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_OFF_POLICY:
                self.ckpt_sync_warper.stop()
        else:
            self.ckpt_sync_warper = NotImplemented
