# 本文件禁止修改，需要修改请联系zhenbinye

from kaiwu_agent.conf import json_strategy
from kaiwu_agent.utils.common_func import simple_factory

def strategy_selector(selector, key, *args, **kwargs):
    """
    参数:
    1. selector:        策略选择器，用于区分策略们，通常是: f"{__name__}.{__class__.__name__}.{sys._getframe().f_code.co_name}"
    2. key:             具体的策略，由配置文件中约定，策略中必须要有一个key表示默认策略，default_strategy
    return: 返回策略结果
    """
    intermediate = json_strategy[selector]
    if hasattr(intermediate, key):
        strategy = intermediate[key]
    else:
        strategy = intermediate['default_strategy']
    return simple_factory(strategy, *args, **kwargs)