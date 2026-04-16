from kaiwu_agent.conf import yaml_metagent
from kaiwu_agent.utils.strategy import strategy_selector 
import sys

def Agent(*args, **kargs):
    return strategy_selector(f"{__name__}.{sys._getframe().f_code.co_name}", yaml_metagent.game_name, *args, **kargs)

def SampleData(*args, **kargs):
    return strategy_selector(f"{__name__}.{sys._getframe().f_code.co_name}", yaml_metagent.game_name, *args, **kargs)

def ObsData(*args, **kargs):
    return strategy_selector(f"{__name__}.{sys._getframe().f_code.co_name}", yaml_metagent.game_name, *args, **kargs)

def ActData(*args, **kargs):
    return strategy_selector(f"{__name__}.{sys._getframe().f_code.co_name}", yaml_metagent.game_name, *args, **kargs)

def observation_process(*args, **kargs):
    return strategy_selector(f"{__name__}.{sys._getframe().f_code.co_name}", yaml_metagent.game_name, *args, **kargs)
    
def action_process(*args, **kargs):
    return strategy_selector(f"{__name__}.{sys._getframe().f_code.co_name}", yaml_metagent.game_name, *args, **kargs)

def sample_process(*args, **kargs):
    return strategy_selector(f"{__name__}.{sys._getframe().f_code.co_name}", yaml_metagent.game_name, *args, **kargs)

def workflow(*args, **kargs):
    return strategy_selector(f"{__name__}.{sys._getframe().f_code.co_name}", yaml_metagent.game_name, *args, **kargs)

# SampleData <----> NumpyData 
def SampleData2NumpyData(*args, **kargs):
    """
        从SampleData解码成NumpyData
    """
    return strategy_selector(f"{__name__}.{sys._getframe().f_code.co_name}", yaml_metagent.game_name, *args, **kargs)
    
def NumpyData2SampleData(*args, **kargs):
    """
        从NumpyData解码成SampleData
    """
    return strategy_selector(f"{__name__}.{sys._getframe().f_code.co_name}", yaml_metagent.game_name, *args, **kargs)

# PBObservation <----> ObsData
def PBObs2ObsData(*args, **kargs):
    """
        编码成obs_data
    """
    return strategy_selector(f"{__name__}.{sys._getframe().f_code.co_name}", yaml_metagent.game_name, *args, **kargs)
    
def ObsData2PBObs(*args, **kargs):
    """
        从obs_data解码成pb_obs
    """
    return strategy_selector(f"{__name__}.{sys._getframe().f_code.co_name}", yaml_metagent.game_name, *args, **kargs)


# PBAction <----> ActData 
def PBAct2ActData(*args, **kargs):
    """
        编码成act_data
    """
    return strategy_selector(f"{__name__}.{sys._getframe().f_code.co_name}", yaml_metagent.game_name, *args, **kargs)
    
def ActData2PBAct(*args, **kargs):
    """
        从act_data解码成pb_act
    """
    return strategy_selector(f"{__name__}.{sys._getframe().f_code.co_name}", yaml_metagent.game_name, *args, **kargs)
    

