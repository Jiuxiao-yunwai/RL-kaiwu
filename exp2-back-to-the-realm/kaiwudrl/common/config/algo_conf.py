#!/usr/bin/env python
# -*- coding: utf-8 -*-


# @file algo_conf.py
# @brief
# @author kaiwu
# @date 2023-11-28


from pydoc import locate
import toml
from kaiwudrl.common.utils.singleton import Singleton
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine


@Singleton
class _AlgoConf:
    """算法配置类, 实例如下:
    normal版本:
    {
    "ppo": {
        "actor_model": "kaiwudrl.common.algorithms.ppo.PPO",
        "learner_model": "kaiwudrl.common.algorithms.ppo.PPO",
        "trainer": "kaiwudrl.server.learner.ppo_trainer.PPOTrainer",
        "predictor": "kaiwudrl.server.actor.ppo_predictor.PPOPredictor",
        "expr_processor": "kaiwudrl.common.algorithms.ppo_processor.PPOProcessor",
        "default_config": "kaiwudrl.common.algorithms.ppo.PPODefaultConfig"
    }
    }

    standard版本:
    dp:
    actor_model: dp.algorithm.agent.Agent
    learner_model: dp.algorithm.agent.Agent
    aisrv_model: dp.algorithm.agent.Agent
    trainer: dp.algorithm.learner.GorgeWalkDPTrainer
    predictor: dp.algorithm.actor.DPPredictor
    train_workflow: dp.train_workflow.workflow
    eval_workflow: kaiwu_agent.gorge_walk.dp.eval_workflow.workflow
    exam_workflow: kaiwu_agent.gorge_walk.dp.exam_workflow.workflow
    """

    class TupleType:
        def __init__(
            self,
            actor_model,
            learner_model,
            trainer,
            predictor,
            expr_processor,
            default_config,
        ):
            self._actor_model = actor_model
            self._learner_model = learner_model
            self._trainer = trainer
            self._predictor = predictor
            # experience processor
            self._expr_processor = expr_processor
            self._default_config = default_config

        def __getattr__(self, key):
            value = super().__getattribute__("_" + key)
            value = locate(value)
            return value

    class StandardTupleType:
        def __init__(
            self,
            actor_model,
            learner_model,
            aisrv_model,
            trainer,
            predictor,
            train_workflow,
            eval_workflow,
            exam_workflow,
        ):
            self._actor_model = actor_model
            self._learner_model = learner_model
            self._aisrv_model = aisrv_model
            self._trainer = trainer
            self._predictor = predictor
            self._train_workflow = train_workflow
            self._eval_workflow = eval_workflow
            self._exam_workflow = exam_workflow

        def __getattr__(self, key):
            value = super().__getattribute__("_" + key)
            value = locate(value)
            return value

    _instance = None

    def __init__(self):
        self.config_map = {
            # 主网络算法，ppo的相关配置
            "ppo": self.TupleType(
                "kaiwudrl.common.algorithms.ppo.PPO",
                "kaiwudrl.common.algorithms.ppo.PPO",
                "kaiwudrl.server.learner.ppo_trainer.PPOTrainer",
                "kaiwudrl.server.actor.ppo_predictor.PPOPredictor",
                "kaiwudrl.common.algorithms.ppo_processor.PPOProcessor",
                "kaiwudrl.common.algorithms.ppo.PPODefaultConfig",
            ),
        }

    def __getitem__(self, key):
        return self.config_map[key]

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def load_conf(self, file_name):
        with open(file_name, "r") as file_obj:
            self._load_conf(file_obj.read())

    def _load_conf(self, algo_str):
        algo_obj = toml.loads(algo_str)

        for algo in algo_obj:
            if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_NORMAL:
                self.config_map[algo] = self.TupleType(**algo_obj[algo])
            elif CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:
                self.config_map[algo] = self.StandardTupleType(**algo_obj[algo])
            else:
                # 未来扩展
                pass


AlgoConf = _AlgoConf()
