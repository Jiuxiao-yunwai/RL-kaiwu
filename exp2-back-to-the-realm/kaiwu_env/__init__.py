from kaiwu_env.utils.strategy import strategy_selector
from kaiwu_env.conf import yaml_arena
from kaiwu_env.utils.common_func import instance_obj_from_file
from common_python.utils.common_func import make_single_dir
import sys
import re
from kaiwu_env.utils.extool import get_global_logger, GlobalMonitor, run_alloc_proxy_proc
from kaiwu_env.utils.extool import reset_global_logger
from kaiwu_env.conf import yaml_logging
import logging
import random
random_generator = random.Random()
if yaml_arena.exam_random_seed != "default":
    random_generator.seed(int(yaml_arena.exam_random_seed))

__doc__ = "This is kaiwu_env, a environment libary for RL study, research or competition"

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# 只启动一个进程, 使用唯一的kaiwu_env.logger, make中有local的logger
logger = logging.getLogger()
# 如果aisvr_type是drl, 需要从七彩石拉取配置写入configure_system.toml文件
if yaml_arena.use_rainbow:
    from kaiwu_env.utils.extool import GlobalRainbow
    rainbow = GlobalRainbow(logger)
    dict_data = rainbow.read_from_rainbow('main')
    rainbow.dump_dict_to_toml_file(dict_data, 'main', 'main_system')

# arena启动需要调用初始化
def setup(**kargs):
    # 额外处理传入参数错误的情况
    if yaml_arena.aisvr_type == 'drl' and 'skylarena_url' in kargs.keys():
        pattern = r'^tcp:\/\/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,5}$'
        if not bool(re.match(pattern, kargs['skylarena_url'])):
            logger.error(f'setup receive a wrong skylarena_url')
            exit()
    # 覆盖arena.yaml 的配置
    for k in kargs.keys():
        if k not in yaml_arena.keys():
            raise KeyError(k)
    yaml_arena.update(kargs)

    # 如果setup改掉默认配置，则要重新设置random_generator
    if yaml_arena.exam_random_seed != "default":
        global random_generator
        random_generator.seed(int(yaml_arena.exam_random_seed))


def make(game_name, scene_name='default_scene', logger=None):
    # 如果是drl使用make, 日志放到目录: '/data/projects' + game_name+'/log/kaiwu_env'
    if logger is None:
        logger = reset_global_logger('/data/projects', game_name+'/log/kaiwu_env', yaml_logging.log_file_prefix+yaml_arena.run_mode+'_')
    else:
        make_single_dir(f"/data/projects/{game_name}/log/kaiwu_env")

    monitor = GlobalMonitor(logger) if yaml_arena.use_prometheus else None
    return run_mode_selector(yaml_arena.run_mode, game_name=game_name, scene_name=scene_name, logger=logger, monitor=monitor)

def scene_selector(game_name, *args, **kargs):
    return strategy_selector("kaiwu_env.__init__.scene_selector", game_name, *args, **kargs)

def run_mode_selector(run_mode, *args, **kargs):
    return strategy_selector("kaiwu_env.__init__.run_mode_selector", run_mode, *args, **kargs)

class StrategySceneSelector:
    @staticmethod
    def gorge_walk(scene_name, logger, monitor):
        from kaiwu_env.gorge_walk.game import Game
        return Game(logger=logger, monitor=monitor)

    @staticmethod
    def back_to_the_realm(scene_name, logger, monitor):
        from sgwrapper import BackToTheRealm
        return BackToTheRealm("btr",600,"map_1")

    @staticmethod
    def back_to_the_realm_rpc(scene_name, logger, monitor):
        if logger:
            logger.info(f"scene_name:{scene_name}, connect to {yaml_arena.rpc_host}:{yaml_arena.rpc_port}")
        return {"rpc_host":yaml_arena.rpc_host, "rpc_port":yaml_arena.rpc_port}

    @staticmethod
    def default_scene(scene_name):
        return "this is default scene"


class StrategyRunModeSelector:
    @staticmethod
    def entity(game_name, scene_name, logger=None, monitor=None):
        env = scene_selector(game_name, scene_name, logger, monitor)
        if yaml_arena.entity_type == "cloak":
            from kaiwu_env.env.game_cloak import GameCloak
            return GameCloak(env, game_name, scene_name, logger=None, monitor=None)
        elif yaml_arena.entity_type == "raw":
            if yaml_arena.comm_type == "lazy":
                from kaiwu_env.env.env_entity_lazy import EnvEntity
                return EnvEntity(env, game_name, scene_name, logger=logger, monitor=monitor)
            elif yaml_arena.comm_type == "busy":
                from kaiwu_env.env.env_entity_busy import EnvEntity
                return EnvEntity(env, game_name, scene_name, logger=logger, monitor=monitor)
            else:
                raise ValueError
        else:
            raise ValueError

    @staticmethod
    def skylarena(game_name, scene_name, logger, monitor):
        if yaml_arena.skylarena_type == "raw":
            if yaml_arena.comm_type == "lazy":
                from kaiwu_env.env.env_skylarena_lazy import EnvSkylarena
                if yaml_arena.use_alloc:
                    run_alloc_proxy_proc()
            elif yaml_arena.comm_type == "busy":
                from kaiwu_env.env.env_skylarena_busy import EnvSkylarena
            else:
                raise ValueError
            return EnvSkylarena(game_name, scene_name, logger, monitor)
        elif yaml_arena.skylarena_type == "2in1":
            env = scene_selector(game_name, scene_name, logger, monitor)
            if yaml_arena.comm_type == "lazy":
                from kaiwu_env.env.env_skylarena_entity_lazy import EnvSkylarenaEntity
            elif yaml_arena.comm_type == "busy":
                from kaiwu_env.env.env_proxy_skylarena_busy import EnvSkylarenaEntity
            else:
                raise ValueError
            return EnvSkylarenaEntity(env, game_name, scene_name, logger, monitor)
        else:
            raise ValueError

    @staticmethod
    def proxy(game_name, scene_name, logger, monitor):
        if yaml_arena.proxy_type == "raw":
            if yaml_arena.comm_type == "lazy":
                from kaiwu_env.env.env_proxy_lazy import EnvProxy
            elif yaml_arena.comm_type == "busy":
                from kaiwu_env.env.env_proxy_busy import EnvProxy
            else:
                raise ValueError
            return EnvProxy(game_name, scene_name, logger, monitor)
        elif yaml_arena.proxy_type == "2in1":
            if yaml_arena.comm_type == "lazy":
                from kaiwu_env.env.env_proxy_skylarena_lazy import EnvProxySkylarena
            elif yaml_arena.comm_type == "busy":
                from kaiwu_env.env.env_proxy_skylarena_busy import EnvProxySkylarena
            else:
                raise ValueError
            return EnvProxySkylarena(game_name, scene_name, logger, monitor)
        else:
            raise ValueError

    @staticmethod
    def default_mode(game_name, scene_name, logger, monitor):
        return "this is default mode"

__all__ = ["make", "logger", "monitor", "random_generator"]

