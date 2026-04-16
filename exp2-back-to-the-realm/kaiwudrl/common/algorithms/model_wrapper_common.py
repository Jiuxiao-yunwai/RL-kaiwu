#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file model_wrapper_common.py
# @brief
# @author kaiwu
# @date 2023-11-28


from kaiwudrl.common.config.algo_conf import AlgoConf
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.algorithms.standard_model_wrapper_builder import (
    StandardModelWrapperBuilder,
)
from kaiwudrl.common.algorithms.model_wrapper_builder import ModelWrapperBuilder
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.common_func import TimeIt
from kaiwudrl.common.checkpoint.model_file_common import process_stop_write_file
from kaiwudrl.common.utils.common_func import get_machine_device_by_config
from kaiwudrl.common.logging.kaiwu_logger import KaiwuLogger, g_not_server_label


def create_standard_model_wrapper(policy_conf, policy_model_wrapper_maps, replay_buffer_wrapper, logger, monitor_proxy):
    """
    建立model_wrapper, 支持多个agent操作, 多个model_wrapper对象, 标准化场景调用
    actor_proxy_local, actor, learner进程会调用
    """

    try:
        # 机器上的device
        machine_device = get_machine_device_by_config(CONFIG.use_which_deep_learning_framework, CONFIG.svr_name)

        # 支持多个agent操作, 此时是多个model_wrapper对象
        for policy_name, policy_conf in policy_conf.items():
            algo = policy_conf.algo
            if CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR or CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
                model = AlgoConf[algo].actor_model(
                    agent_type=CONFIG.svr_name,
                    device=machine_device,
                    logger=logger,
                    monitor=monitor_proxy if CONFIG.use_prometheus else None,
                )
            elif CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
                model = AlgoConf[algo].learner_model(
                    agent_type=CONFIG.svr_name,
                    device=machine_device,
                    logger=logger,
                    monitor=monitor_proxy if CONFIG.use_prometheus else None,
                )
            else:
                continue

            model_wrapper = StandardModelWrapperBuilder().create_model_wrapper(model, logger)
            model.framework_handler = model_wrapper
            # self.workflow = AlgoConf[CONFIG.algo].actor_workflow()

            if KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX == CONFIG.use_which_deep_learning_framework:
                model_wrapper.build_predict_graph(input_tensors)
                model_wrapper.add_predict_hooks(predict_hooks())
                model_wrapper.create_predict_session()

                global_step = model_wrapper.get_global_step()

            elif KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE == CONFIG.use_which_deep_learning_framework:
                # 注意单机单进程可能遇见
                if CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
                    model_wrapper.set_dataset(replay_buffer_wrapper)
                model_wrapper.build_model()

            elif KaiwuDRLDefine.MODEL_PYTORCH == CONFIG.use_which_deep_learning_framework:
                # 注意单机单进程可能遇见
                if CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
                    model_wrapper.set_dataset(replay_buffer_wrapper)

            elif KaiwuDRLDefine.MODEL_TCNN == CONFIG.use_which_deep_learning_framework:
                pass

            elif KaiwuDRLDefine.MODEL_TENSORRT == CONFIG.use_which_deep_learning_framework:
                # 注意单机单进程可能遇见
                if CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
                    model_wrapper.set_dataset(replay_buffer_wrapper)
                model_wrapper.build_model()

            else:
                logger.error(
                    f"error use_which_deep_learning_framework "
                    f"{CONFIG.use_which_deep_learning_framework}, only support "
                    f"{KaiwuDRLDefine.MODEL_TCNN}, {KaiwuDRLDefine.MODEL_PYTORCH}, "
                    f"{KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX}, "
                    f"{KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE}",
                    g_not_server_label,
                )

                continue

            policy_model_wrapper_maps[policy_name] = model_wrapper

            logger.info(f"policy_name {policy_name}, algo {algo}, model_wrapper is {model_wrapper.name}")

    except Exception as e:
        logger.exception(f" failed to run create_normal_model_wrapper. exit. Error is: {e}, ")

        # 报错后让其提前退出去
        error_code = KaiwuDRLDefine.DOCKER_EXIT_CODE_ERROR
        process_stop_write_file(error_code, logger)


def create_normal_model_wrapper(policy_conf, policy_model_wrapper_maps, replay_buffer_wrapper, logger):
    """
    建立model_wrapper, 支持多个agent操作, 多个model_wrapper对象, 非标准化场景调用
    actor_proxy_local, actor, learner会调用
    """

    try:
        with TimeIt() as ti:
            # 支持多个agent操作, 此时是多个model_wrapper对象
            for policy_name, policy_conf in policy_conf.items():
                algo = policy_conf.algo

                # 区分不同的actor, learner, aisrv调用
                networks = None
                if CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR or CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
                    networks = policy_conf.actor_network
                elif CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
                    networks = policy_conf.learner_network
                else:
                    pass

                if not networks:
                    logger.error(f"networks is None, policy_name is {policy_name}, please check")
                    return

                # network, 兼顾多个network
                if len(networks) == 1:
                    network = networks[0](
                        policy_conf.state.state_space(),
                        policy_conf.action.action_space(),
                    )
                else:
                    network = [
                        net(
                            policy_conf.state.state_space(),
                            policy_conf.action.action_space(),
                        )
                        for net in networks
                    ]

                # model
                name = "%s_%s" % (CONFIG.app, algo)
                if CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR or CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
                    model = AlgoConf[algo].actor_model(network, name, CONFIG.svr_name)
                elif CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
                    model = AlgoConf[algo].learner_model(network, name, CONFIG.svr_name)
                else:
                    continue

                model_wrapper = ModelWrapperBuilder().create_model_wrapper(model, logger)

                if KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX == CONFIG.use_which_deep_learning_framework:
                    model_wrapper.build_predict_graph(input_tensors)
                    model_wrapper.add_predict_hooks(predict_hooks())
                    model_wrapper.create_predict_session()

                    global_step = model_wrapper.get_global_step()

                elif KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE == CONFIG.use_which_deep_learning_framework:
                    # 注意单机单进程可能遇见
                    if CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
                        model_wrapper.set_dataset(replay_buffer_wrapper)

                    model_wrapper.build_model()

                elif KaiwuDRLDefine.MODEL_PYTORCH == CONFIG.use_which_deep_learning_framework:
                    # 注意单机单进程可能遇见
                    if CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
                        model_wrapper.set_dataset(replay_buffer_wrapper)

                elif KaiwuDRLDefine.MODEL_TCNN == CONFIG.use_which_deep_learning_framework:
                    pass

                elif KaiwuDRLDefine.MODEL_TENSORRT == CONFIG.use_which_deep_learning_framework:
                    # 注意单机单进程可能遇见
                    if CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
                        model_wrapper.set_dataset(replay_buffer_wrapper)

                    model_wrapper.build_model()

                else:
                    logger.error(
                        f"error use_which_deep_learning_framework "
                        f"{CONFIG.use_which_deep_learning_framework}, only support "
                        f"{KaiwuDRLDefine.MODEL_TCNN}, {KaiwuDRLDefine.MODEL_PYTORCH}, "
                        f"{KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX}, "
                        f"{KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE}",
                        g_not_server_label,
                    )

                    continue

                policy_model_wrapper_maps[policy_name] = model_wrapper

                logger.info(f"policy_name {policy_name}, algo {algo}, model_wrapper is {model_wrapper.name}")

    except Exception as e:
        logger.exception(f" failed to run create_normal_model_wrapper. exit. Error is: {e}, ")

        # 报错后让其提前退出去
        error_code = KaiwuDRLDefine.DOCKER_EXIT_CODE_ERROR
        process_stop_write_file(error_code, logger)
