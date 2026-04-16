#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @file online_rl_helper.py
# @brief
# @author kaiwu
# @date 2023-11-28


import time
from kaiwudrl.server.aisrv.kaiwu_rl_helper import KaiWuRLHelper
from kaiwudrl.interface.exception import (
    SkipEpisodeException,
    ClientQuitException,
    TimeoutEpisodeException,
)
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.common_func import TimeIt
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine


class OnLineRLHelper(KaiWuRLHelper):
    def run(self) -> None:
        try:
            self.env.init()
        except AssertionError as e:
            self.logger.exception(f"kaiwu_rl_helper self.env.init() {str(e)}")
            self.env.reject(e)
        except Exception as e:
            self.logger.exception(f"kaiwu_rl_helper self.env.init() {str(e)}")
            self.env.reject(e)
        else:
            self.client_id = self.env.client_id
            try:
                self.agent_ctxs = {}
                self.simu_ctx.agent_ctxs = self.agent_ctxs

                def run_episode_once():
                    self.run_episode()

                while not self.exit_flag.value:
                    try:
                        run_episode_once()

                    except SkipEpisodeException:
                        self.logger.error(
                            "kaiwu_rl_helper run_episode_once() SkipEpisodeException {}",
                            str(e),
                        )
                        pass
            except ClientQuitException:
                self.logger.exception(f"kaiwu_rl_helper run_episode_once() ClientQuitException {str(e)}, ")
            except TimeoutEpisodeException as e:
                self.logger.exception(f"kaiwu_rl_helper run_episode_once() TimeoutEpisodeException {str(e)}, ")
                self.env.reject(e)
            except AssertionError as e:
                self.logger.exception(f"kaiwu_rl_helper run_episode_once() AssertionError {str(e)}, ")
                self.env.reject(e)
            except Exception as e:
                self.logger.exception(f"kaiwu_rl_helper run_episode_once() Exception {str(e)}, ")
                if not self.exit_flag.value:
                    self.env.reject(e)
        finally:
            self.logger.info("kaiwu_rl_helper finally")
            self.stop()

    def run_episode(self):

        with TimeIt() as ti:
            self.online_episode_main_loop()

    def online_episode_main_loop(self):
        """
        业务主函数

        监控训练指标定义如下:
        业务会在aisrv里上报自定义的监控指标, 故这里增加上, map形式, 由业务自己定义
        self.app_monitor_data = {}
        self.app_monitor_data['key1'] = 1
        self.app_monitor_data['key2'] = 2
        self.app_monitor_data['key3'] = 3
        """

        counter = 0
        states, must_need_sample_info = self.env.next_valid()
        """
        if not states:
            raise
        """
        last_format_action_list = [-1000, -1000]
        while not self.exit_flag.value:
            try:
                valid_agents = list(states.keys())
                # 如果没有初始化，则进行初始化
                for agent_id in valid_agents:
                    if agent_id not in self.agent_ctxs:
                        self.start_agent(agent_id)

                    agent_ctx = self.agent_ctxs[agent_id]
                    agent_ctx.state, agent_ctx.pred_input = {}, {}

                    policy_id = agent_ctx.main_id
                    s = states[agent_id].get_state()
                    agent_ctx.pred_input[policy_id] = s
                    agent_ctx.state[policy_id] = states[agent_id]

                # 执行预测
                self.predict(valid_agents)
                self.logger.debug("kaiwu_rl_helper predict success")

                # 解析action
                format_action_list = []
                for agent_id in valid_agents:
                    agent_ctx = self.agent_ctxs[agent_id]
                    for policy_id in agent_ctx.policy:
                        format_action = agent_ctx.pred_output[policy_id][agent_id]["format_action"]
                        network_sample_info = agent_ctx.pred_output[policy_id][agent_id]["network_sample_info"]
                        # lstm_info = agent_ctx.pred_output[policy_id][agent_id]['lstm_info']
                        format_action_list.append(format_action)

                        self.from_actor_model_version = agent_ctx.pred_output[policy_id][agent_id]["model_version"]

                # 评估模式规则, 走重复的路采用随机动作
                if False and CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EVAL:
                    for i in range(len(format_action_list)):
                        while (
                            abs(format_action_list[i] - last_format_action_list[i]) == 1
                            and (format_action_list[i] + last_format_action_list[i]) != 3
                        ):
                            act = np.random.choice(range(DimConfig.DIM_OF_ACTION), 1).tolist()[0]
                            format_action_list[i] = act

                # __gorge_walk_step
                self.env.on_handle_action(format_action_list)
                _states, must_need_sample_info = self.env.next_valid()

                # 再次执行一次预测, 得到next_action (td_sarsa需要next_action)
                if not self.env.run_handler.done and CONFIG.algo == "td_sarsa":
                    s = _states[agent_id].get_state()
                    agent_ctx.pred_input[policy_id] = s
                    agent_ctx.state[policy_id] = _states[agent_id]
                    self.predict(valid_agents)

                    # 解析next_action
                    next_format_action_list = []
                    for agent_id in valid_agents:
                        agent_ctx = self.agent_ctxs[agent_id]
                        for policy_id in agent_ctx.policy:
                            next_format_action = agent_ctx.pred_output[policy_id][agent_id]["format_action"]
                            next_format_action_list.append(next_format_action)
                else:
                    next_format_action_list = [[-1]]

                # 存储和发送样本的agent_id和policy_id
                agent_id = valid_agents[0]
                agent_ctx = self.agent_ctxs[agent_id]
                policy_id = agent_ctx.main_id

                # 存储样本
                if agent_ctx.policy[policy_id].need_train():
                    self.gen_expr(
                        valid_agents[0],
                        self.agent_ctxs[valid_agents[0]].main_id,
                        {
                            "must_need_sample_info": {
                                "last_state": states,
                                "state": _states,
                                "action": format_action_list,
                                "info": must_need_sample_info,
                                "next_action": next_format_action_list,
                            },
                            "network_sample_info": {
                                "log_prob": network_sample_info[0],
                                "value": network_sample_info[1],
                            },
                        },
                    )

                # 满足条件发送样本
                if (counter + 1) % int(CONFIG.send_sample_size) == 0 and not self.env.run_handler.done:
                    for policy_id in agent_ctx.policy:
                        if agent_ctx.policy[policy_id].need_train():
                            self.gen_train_data(agent_id, policy_id)

                # 在on-policy的情况下, 如果按照每帧执行时执行on-policy的流程, 需要暂停该线程处理
                if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                    if CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_STEP:
                        self.process_pause()
                        self.logger.info("kaiwu_rl_helper pause when episode or step reach under on-policy")

                        # 如果should_stop设置为True说明此时需要同步等待
                        while self.get_process_in_pause_statues():
                            time.sleep(CONFIG.idle_sleep_second)

                        self.logger.info("kaiwu_rl_helper continue when episode or step reach under on-policy")

                # 处理游戏结束信号
                if self.env.run_handler.done:
                    self.logger.info("kaiwu_rl_helper game is over")
                    # 样本生成器是单例模式，里面包含所有Agent生成的样本，因此只需要调用一次
                    for policy_id in agent_ctx.policy:
                        if agent_ctx.policy[policy_id].need_train():
                            self.gen_train_data(agent_id, policy_id)

                        self.logger.debug("kaiwu_rl_helper gen_train_data success")

                    # 在on-policy的情况下, 如果是按照每局结束时执行on-policy流程, 需要暂停该线程处理
                    if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                        if CONFIG.on_policy_by_way == KaiwuDRLDefine.ALGORITHM_ON_POLICY_WAY_EPISODE:
                            self.process_pause()
                            self.logger.info("kaiwu_rl_helper pause when episode or step reach under on-policy")

                            # 如果should_stop设置为True说明此时需要同步等待
                            while self.get_process_in_pause_statues():
                                time.sleep(CONFIG.idle_sleep_second)

                            self.logger.info("kaiwu_rl_helper continue when episode or step reach under on-policy")

                    self.stop_agent(agent_id)
                    # 游戏结束后，不需要预测，但是需要返回一个空的action
                    self.env.on_handle_action([[0]])

                    # 主动退出循环, 框架需要的操作
                    self.exit_flag.value = True
                    # 处理异常退出情况,保存样本
                    if not self.env.run_handler.done:
                        # 样本生成器是单例模式，里面包含所有Agent生成的样本，因此只需要调用一次
                        pass

                    # 结束aisrv的循环
                    self.env.msg_buff.output_q.put(None)
                    break

                # 更新当前state，准备下一轮迭代
                states = _states
                last_format_action_list = format_action_list
                counter += 1

            except AssertionError as e:
                self.logger.exception(f"kaiwu_rl_helper sgame_episode_main_loop {str(e)}, ")
                self.env.reject(e)
                break
            except ClientQuitException as e:
                self.logger.exception(f"kaiwu_rl_helper sgame_episode_main_loop ClientQuitException {str(e)}, ")
                break
            except TimeoutEpisodeException as e:
                self.logger.exception(f"kaiwu_rl_helper sgame_episode_main_loop TimeoutEpisodeException {str(e)}, ")
                self.env.reject(e)
                break
            except Exception as e:
                self.logger.exception(f"kaiwu_rl_helper sgame_episode_main_loop {str(e)}, ")
                self.env.reject(e)
                break
