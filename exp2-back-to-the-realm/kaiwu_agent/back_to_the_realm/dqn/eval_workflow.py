from kaiwu_agent.agent.protocol.protocol import *
from kaiwu_agent.conf import yaml_back_to_the_realm_game as game_conf
from agent_dqn.feature.preprocessor import Preprocessor

# from tools.train_env_conf_validate import check_usr_conf, read_usr_conf


def workflow(envs, agents, logger=None, monitor=None):
    env, agent = envs[0], agents[0]

    treasure_count = game_conf.treasure_count
    treasure_random = game_conf.treasure_random
    max_step = game_conf.max_step

    if treasure_count is None or treasure_count < 0:
        treasure_count = 0
    list_treasure_id = [x for x in range(3, treasure_count + 3)]

    # 评估开始
    logger.info(".......... Evaluation Start ..........")

    # # 配置文件读取和校验
    # usr_conf = read_usr_conf("agent_dqn/conf/train_env_conf.toml", logger)
    # if usr_conf is None:
    #     logger.error(f"usr_conf is None, please check agent_dqn/conf/train_env_conf.toml")
    #     raise ValueError("usr_conf is None, please check agent_dqn/conf/train_env_conf.toml")
    # valid = check_usr_conf(usr_conf, logger)
    # if not valid:
    #     logger.error(f"check_usr_conf return False, please check agent_dqn/conf/train_env_conf.toml")
    #     raise ValueError("check_usr_conf return False, please check agent_dqn/conf/train_env_conf.toml")

    # # 打印下usr_conf
    # logger.info(f"usr_conf is {usr_conf}")

    EPISODE_CNT = 1
    total_score, win_cnt, treasure_cnt = 0, 0, 0
    episode = 0
    while episode < EPISODE_CNT:
        # 用户自定义的游戏启动配置
        usr_conf = {
            "env_conf": {
                "start": 2,
                "end": 1,
                "treasure_id": list_treasure_id,
                "treasure_random": treasure_random,
                "treasure_count": treasure_count,
                "talent_type": 1,
                "max_step": max_step,
            }
        }

        # 打印下usr_conf
        logger.info(f"usr_conf is {usr_conf}")

        # 特征值处理
        feature_process = Preprocessor()

        # 重置游戏, 并获取初始状态
        obs, state = env.reset(usr_conf=usr_conf)

        # 特征处理
        obs_data, remain_info = agent.observation_process(obs, feature_process, state)

        # 游戏循环
        done = False
        current_score = 0
        while not done:
            # Agent 进行推理, 获取下一帧的预测动作
            act_data, model_version = agent.exploit(list_obs_data=[obs_data])

            # ActData 解包成动作
            act = agent.action_process(act_data[0])

            # 与环境交互, 执行动作, 获取下一步的状态
            frame_no, _obs, score, terminated, truncated, _state = env.step(act)
            if _obs == None:
                logger.info(f"env.step return None, so break")
                episode -= 1
                current_score = 0
                break

            # 特征处理
            _obs_data, _remain_info = agent.observation_process(_obs, feature_process, _state)

            # 判断游戏结束, 并更新胜利次数
            done = terminated or truncated
            logger.info(f"env.step return done, terminated: {terminated}, truncated: {truncated}, so break")
            if terminated:
                win_cnt += 1

            # 更新总奖励和状态
            current_score = score.total_score
            obs_data = _obs_data

        # 更新宝箱收集数量
        treasure_cnt += _obs.game_info.treasure_count

        # 更新总奖励和状态
        total_score += current_score

        episode += 1

    # 打印评估结果
    logger.info(f"Average Total Score: {total_score / EPISODE_CNT}")
    logger.info(f"Average Treasure Collected: {treasure_cnt / EPISODE_CNT}")
    logger.info(f"Success Rate : {win_cnt / EPISODE_CNT}")

    # 评估结束
    logger.info(".......... Evaluation End ..........")
