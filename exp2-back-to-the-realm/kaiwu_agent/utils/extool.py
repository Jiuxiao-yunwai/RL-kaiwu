
import kaiwu_agent
import datetime, os
from kaiwu_agent.utils.common_func import Singleton
from kaiwu_agent.conf import yaml_logging
from kaiwu_agent.conf import yaml_alloc
from kaiwu_agent.conf import yaml_monitor
from kaiwu_agent.conf import yaml_rainbow
from common_python.logging.kaiwu_logger import KaiwuLogger
from common_python.alloc.alloc_proxy import AllocProxy
from common_python.alloc.alloc_utils import AllocUtils
from common_python.monitor.monitor_proxy import MonitorProxy
from common_python.config.config_control import CONFIG
import logging
import yaml

def get_global_logger(log_file_prefix=None):
    # 若只启动一个进程, 使用唯一的logger
    logger = KaiwuLogger()
    log_file_prefix = yaml_logging.log_file_prefix if log_file_prefix == None else log_file_prefix
    logger.setLoggerFormat(f"{yaml_logging.log_dir}/{yaml_logging.svr_name}/{log_file_prefix}{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", params=yaml_logging)
    return logger

def reset_global_logger(logger, log_dir, svr_name, log_file_prefix=None):
    log_file_prefix = yaml_logging.log_file_prefix if log_file_prefix == None else log_file_prefix
    logger.setLoggerFormat(f"{log_dir}/{svr_name}/{log_file_prefix}{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", params=yaml_logging)

def run_alloc_proxy_proc(file_path=None, section=None):
    file_path = os.path.join(kaiwu_agent.__path__[0], yaml_alloc.file_path) if file_path == None else file_path
    section = yaml_alloc.section if section == None else section
    # CONFIG.parse_configure(['main_system'], '/data/projects/Metagent/arena/conf/configure_system.toml')
    CONFIG.parse_configure([section], file_path)
    alloc_proxy = AllocProxy()
    alloc_proxy.daemon = True
    alloc_proxy.start()

def get_ip_by_alloc(srv_name, target_role, count, file_path=None, section=None, logger=None):
    file_path = os.path.join(kaiwu_agent.__path__[0], yaml_alloc.file_path) if file_path == None else file_path
    section = yaml_alloc.section if section == None else section
    logger = logger if logger else logging.getLogger(__name__)
    CONFIG.parse_configure([section], file_path)

    alloc_utils = AllocUtils(logger)

    params = {
            'role' : CONFIG.alloc_process_role,
            'set_name' : CONFIG.set_name,
            'task_id' : CONFIG.task_id,
            'alloc_process_address' : CONFIG.alloc_process_address,
            'assign_limit' : CONFIG.alloc_process_assign_limit,
            'port' : CONFIG.port
        }
    alloc_utils.set_params(params)

    # 注册函数
    code, message = alloc_utils.registry()
    if code:
        logger.info(f'registry success')
    else:
        logger.error(f'registry failed, error is {message}')

    ip_address = alloc_utils.get_ip_by_server(srv_name, CONFIG.set_name, target_role, count)
    if ip_address:
        logger.info(f'get ip success, ip is {ip_address}')
    else:
        logger.error(f'get ip failed')
    return ip_address

@Singleton
class GlobalMonitor:
    def __init__(self, logger, file_path=None, section=None):
        file_path = os.path.join(kaiwu_agent.__path__[0], yaml_monitor.file_path) if file_path == None else file_path
        section = yaml_monitor.section if section == None else section
        CONFIG.parse_configure([section], file_path)
        self.monitor_proxy = MonitorProxy(logger)
        self.monitor_proxy.daemon = True
        self.monitor_proxy.start()

    def put_data(self, monitor_data):
        self.monitor_proxy.put_data(monitor_data)

@Singleton
class GlobalRainbow:
    def __init__(self, logger, file_path=None, section=None):
        from common_python.utils.rainbow_utils import RainbowUtils
        file_path = os.path.join(kaiwu_agent.__path__[0], yaml_rainbow.file_path) if file_path == None else file_path
        section = yaml_rainbow.section if section == None else section
        CONFIG.parse_configure([section], file_path, ['main_system'], file_path)
        self.rainbow_utils = RainbowUtils(CONFIG.rainbow_url, CONFIG.rainbow_app_id, CONFIG.rainbow_user_id, CONFIG.rainbow_secret_key, CONFIG.rainbow_env_name, logger)
        self.logger = logger

    def read_from_rainbow(self, rainbow_group):
        # rainbow_group = 'main' or 'skylarena' or 'game' and so on
        result_code, data, result_msg = self.rainbow_utils.read_from_rainbow(rainbow_group)
        if result_code:
            self.logger.warning(f'read_from_rainbow failed, err msg is {result_msg}')
            return False
        return data

    def dump_dict_to_yaml_file(self, dict_source):
        # 此函数还未完善
        yaml_rainbow.render_config_from_dict(dict_source)
        yaml_rainbow.dump_config_to_file(yaml_rainbow.config_file_path)

    def dump_dict_to_toml_file(self, dict_source, process_name, targe_section):
        to_change_key_values = yaml.load(dict_source[process_name], Loader=yaml.SafeLoader)
        CONFIG.write_to_config(to_change_key_values)
        CONFIG.save_to_file(targe_section, to_change_key_values)
