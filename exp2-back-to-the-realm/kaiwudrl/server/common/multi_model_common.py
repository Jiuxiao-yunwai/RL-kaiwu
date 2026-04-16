#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @file multi_model_common.py
# @brief
# @author kaiwu
# @date 2023-11-28


import os
import re
import shutil
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.common_func import (
    TimeIt,
    make_single_dir,
    python_exec_shell,
    get_machine_device_by_config,
)
import importlib.util
import importlib.machinery
import sys
import json


class MultiModelManager(object):
    """
    该类主要是用于管理对战模型里使用, 比如最新模型需要与旧模型对战, 包括网络参数, 特征处理, 网络结构等都需要采用当时模型包里设置的内容
    """

    def __init__(self, policy_name, monitor_proxy, logger) -> None:
        self.logger = logger
        self.monitor_proxy = monitor_proxy
        self.policy_name = policy_name

        # 模型ID --> 模型的映射
        self.model_id_to_models = {}

        # 由于预测进程可能有多个, 这里拷贝和解压缩都需要加上预测进程的pid
        self.pid = os.getpid()

        # 加载预先设置的模型
        self.init_load_models()

    def copy_model_files(self):
        """
        拷贝model文件, 主要包括下面操作:
        1. 在项目主目录下新建以model_id为名字的目录
        2. 拷贝原始的model文件包到model_id目录下
        3. 解压缩zip
        4. 删除掉zip包, 可选, 代码稳定前不建议删除
        """
        source_model_dir_list = CONFIG.init_model_file_list.split(",")
        if not source_model_dir_list:
            self.logger.info(f"predict init_model_file_list is empty, so return")
            return None

        source_model_id_list = CONFIG.init_model_file_id_list.split(",")
        if not source_model_id_list:
            self.logger.info(f"predict init_model_file_id_list is empty, so return")
            return None

        if len(source_model_dir_list) != len(source_model_id_list):
            self.logger.info(
                f"predict len(source_model_dir_list) {len(source_model_dir_list)}"
                f"!= len(source_model_id_list) {len(source_model_id_list)}, so return"
            )
            return None

        make_single_dir(CONFIG.model_pools_dir)

        # 当前代码包目录
        current_dir = f"{CONFIG.model_pools_dir}/{self.pid}"
        make_single_dir(current_dir)
        # 写入空的__init__.py文件
        with open(f"{current_dir}/__init__.py", "w") as file:
            pass

        count = 0
        # 拷贝model文件包
        for i in range(len(source_model_dir_list)):
            model_file = source_model_dir_list[i].strip()
            if os.path.exists(model_file):
                if count > CONFIG.init_model_file_list_max:
                    self.logger.info(
                        f"predict init_model_file_list count > {CONFIG.init_model_file_list_max}, so return"
                    )
                    break

                model_dir = f"{current_dir}/model_{source_model_id_list[i].strip()}_pid_{self.pid}"
                make_single_dir(model_dir)
                # 写入空的__init__.py文件
                with open(f"{model_dir}/__init__.py", "w") as file:
                    pass

                shutil.copy(model_file, model_dir)
                count += 1
                self.logger.info(f"predict {model_file} copy to {current_dir} success")

        # 返回的数据结构
        result = {}

        # 解压缩zip包
        for model_id in source_model_id_list:
            if model_id:
                # 注意区分单个机器上不同的进程都需要加载该model文件
                model_id_dir = f"{current_dir}/model_{model_id}_pid_{self.pid}"

                shell_content = f"cd {model_id_dir}; unzip -o {model_id}.zip; touch __init__.py; cd -"
                python_exec_shell(shell_content)
                self.logger.info(f"predict {shell_content} success")

                # 确认模型包下采用的算法, 采用读取ckpt/kaiwu.json而不是conf/configure_app.toml
                config_file_path = f"{model_id_dir}/ckpt/kaiwu.json"
                if not os.path.exists(config_file_path):
                    self.logger.error(f"predict {config_file_path} not exist, please check")
                    continue

                # 读取配置文件
                with open(config_file_path, "r") as config_file:
                    config_data = json.load(config_file)

                # 获取 algo 的值
                algo_value = config_data.get("algorithm", None)
                if algo_value is None:
                    self.logger.error(f"predict algo_value is None, please check")
                    continue

                self.logger.info(f"predict algo_value is {algo_value}")

                """
                去掉装饰器, 否则调用会出现问题
                因为存在需要兼容的场景, 故这里先检测agent.py是否存在, 前提是保证2种场景下的项目是不存在的即可
                """
                python_file_path = f"{model_id_dir}/agent_{algo_value}/agent.py"
                if not os.path.exists(python_file_path):
                    python_file_path = f"{model_id_dir}/{algo_value}/algorithm/agent.py"
                    algo_name = algo_value
                else:
                    algo_name = f"agent_{algo_value}"

                if not self.remove_the_decorator(python_file_path):
                    self.logger.error(f"predict self.remove_the_decorator is failed, please check")
                    continue

                self.logger.info(f"predict remove_the_decorator success, python_file_path is {python_file_path}")

                # 修改导入路径为相对路径
                self.change_import_to_relative(model_id_dir, model_id, algo_name)

                info = {}
                info["algo"] = algo_value
                info["model_path"] = f"{model_id_dir}/ckpt"

                result[int(model_id)] = info

        return result

    def remove_the_decorator(self, python_file_path):
        """
        去掉文本里的装饰器和函数
        """
        if not python_file_path or not os.path.exists(python_file_path):
            self.logger.error(f"predict python_file_path {python_file_path} not exist, please check")
            return False

        # 要删除的装饰器列表和函数
        decorators_to_remove = [
            "@predict_wrapper",
            "@exploit_wrapper",
            "@load_model_wrapper",
            "@learn_wrapper",
            "@save_model_wrapper",
            "torch.set_num_interop_threads(1)",
        ]

        # 读取文件内容
        with open(python_file_path, "r") as file:
            lines = file.readlines()

        # 移除指定的装饰器
        with open(python_file_path, "w") as file:
            for line in lines:
                # 检查当前行是否包含需要删除的装饰器
                if any(decorator in line for decorator in decorators_to_remove):
                    # 如果是，跳过这一行
                    continue

                # 否则，将行写入文件
                file.write(line)

        return True

    def change_import_to_relative(self, project_root, model_id, algo_name):

        # 正则表达式匹配绝对导入语句
        import_pattern = re.compile(r"^(from\s+{}\.|import\s+{}\.)".format(algo_name, algo_name))
        replace_name = f"model_{model_id}_pid_{self.pid}.{algo_name}"

        # 遍历目录下的所有文件
        for root, dirs, files in os.walk(project_root):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r") as f:
                        file_contents = f.readlines()

                    # 处理每一行
                    for i, line in enumerate(file_contents):
                        match = import_pattern.match(line)
                        if match:
                            # 获取导入的模块路径和导入的项
                            parts = line.strip().split()
                            module_path = parts[1].replace(f"{algo_name}.", "")
                            # 重构导入语句
                            if parts[0] == "import":
                                file_contents[i] = f"import {replace_name}.{module_path}\n"
                            elif parts[0] == "from":
                                import_items = " ".join(parts[3:])
                                file_contents[i] = f"from {replace_name}.{module_path} import {import_items}\n"

                    # 写回修改后的文件内容
                    with open(file_path, "w") as f:
                        f.writelines(file_contents)

        self.logger.info(
            f"predict All absolute imports have been converted to relative imports, "
            f"project_root is {project_root}, algo_name is {algo_name}"
        )

    def init_load_models(self):
        """
        初始化model调用, 主要操作如下:
        1. 根据配置conf/kaiwudrl/configure.toml文件里的配置项init_model_file_list将文件从目录拷贝到项目主目录下
        2. 调用MultiModel类加载Agent
        """
        result = self.copy_model_files()
        if result is None or not result:
            self.logger.info(f"predict just no model_pool files or copy_model_files failed, please check")
            return

        for model_id, model_info in result.items():
            multi_model = MultiModel(
                self.policy_name,
                model_id,
                model_info.get("algo"),
                model_info.get("model_path"),
                self.monitor_proxy,
                self.logger,
            )
            if not multi_model.init():
                self.logger.error(f"predict multi_model.init failed, please check")
                continue

            self.model_id_to_models[str(model_id)] = multi_model

    def predict(self, model_id, predict_data):
        """
        预测函数
        """
        if model_id is None:
            return None

        model_object = self.model_id_to_models.get(str(model_id), None)
        if not model_object:
            return None

        return model_object.predict(predict_data)

    def exploit(self, model_id, predict_data):
        """
        利用函数
        """
        if model_id is None:
            return None

        model_object = self.model_id_to_models.get(str(model_id), None)
        if not model_object:
            return None

        return model_object.exploit(predict_data)

    def load_model(self, model_id):
        """
        加载模型函数
        """
        if model_id is None:
            return None

        model_object = self.model_id_to_models.get(str(model_id), None)
        if not model_object:
            return None

        return model_object.load_model()

    def get_model_path(self, model_id):
        """
        返回model文件所在的目录
        """
        if model_id is None:
            return None

        model_object = self.model_id_to_models.get(str(model_id), None)
        if not model_object:
            return None

        return model_object.get_model_path()

    def get_predict_stat(self):
        """
        统计预测次数
        """
        predict_success_count = 0
        for model_id, model_object in self.model_id_to_models.items():
            predict_success_count += model_object.get_predict_stat()

        return predict_success_count


class MultiModel(object):
    """
    该类主要处理单个模型包内容, 主要是当时的模型包内容, 模型包内容如下:
    1. conf, 配置文件
    2. dqn, 算法包
    3. ckpt模型内容

    需要做的操作:
    1. 进程开始时加载当时的模型文件
    2. 预测和利用使用当时的代码包内容
    """

    def __init__(self, policy_name, model_id, algo, model_path, monitor_proxy, logger) -> None:
        self.model_id = model_id
        self.policy_name = policy_name
        self.algo = algo
        self.logger = logger
        self.monitor_proxy = monitor_proxy

        # 智能体
        self.agent = None
        self.model_path = model_path
        self.ckpt_id = self.get_ckpt_id()

        self.pid = os.getpid()

        # 统计使用
        self.predict_success_count = 0

    def get_ckpt_id(self):
        """
        获取具体的ckpt_id
        """
        kaiwu_json_file = f"{self.model_path}/kaiwu.json"
        with open(kaiwu_json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get("train_step", 0)

    def load_model(self):
        """
        加载指定的model文件
        """
        try:
            self.agent.load_model(self.model_path, self.ckpt_id)
            self.logger.info(f"predict load_model success, {self.model_path}, {self.ckpt_id}")

        except Exception as e:
            self.logger.error(f"predict self.agent.load_model failed, error is {str(e)}, please check")
            return False

        return True

    def get_model_path(self):
        """
        返回model文件所在的目录
        """
        return self.model_path

    def make_agent(self):
        """
        使用的Agent类:
        1. 旧版本, 类似/data/ckpt/149324/diy/algorithm/agent.py里有Agent
        2. 新版本, 类似/data/ckpt/149324/agent_diy/agent.py里有Agent

        采用的方法是判断文件是否存在, 前提是不能新旧版本都存在一个项目里
        """

        # 保存原始的sys.path
        original_sys_path = list(sys.path)
        try:
            sys.path.insert(0, f"{CONFIG.model_pools_dir}/{self.pid}/")

            module_name = f"agent_module_{self.policy_name}_{self.model_id}"
            module_path = (
                f"{CONFIG.model_pools_dir}/{self.pid}/model_{self.model_id}_pid_{self.pid}/agent_{self.algo}/agent.py"
            )
            if not os.path.exists(module_path):
                module_path = f"{CONFIG.model_pools_dir}/{self.pid}/model_{self.model_id}_pid_{self.pid}/{self.algo}/algorithm/agent.py"

            spec = importlib.util.spec_from_file_location(module_name, module_path)

            # 创建一个新的模块基于这个规范
            module = importlib.util.module_from_spec(spec)
            # 执行模块的代码
            spec.loader.exec_module(module)
            Agent = getattr(module, "Agent")

            # 创建 Agent 类的实例
            machine_device = get_machine_device_by_config(CONFIG.use_which_deep_learning_framework, CONFIG.svr_name)
            self.agent = Agent(
                agent_type="player", device=machine_device, logger=self.logger, monitor=self.monitor_proxy
            )
            self.logger.info(f"predict self.agent {self.agent} make success")

            if self.agent is None:
                self.logger.error(f"predict self.agent is None, please check")
                return False
        except Exception as e:
            self.logger.exception(f"predict make_agent failed, error msg is {str(e)},")
            return False
        finally:
            # 恢复原始的 sys.path
            sys.path = original_sys_path

        return True

    def init(self):
        """
        主要是下面的操作:
        1. 构建Agent, 采用的是当时的agent.py里的类
        2. 加载model文件, 采用的是当时的model文件
        """
        # 构建Agent
        if not self.make_agent():
            self.logger.error(f"predict self.make_agent failed, please check")
            return False

        # 加载model文件
        if not self.load_model():
            self.logger.error(f"predict self.load_model failed, please check")
            return False

        return True

    def predict(self, predict_data):
        """
        调用业务的predict函数
        """
        if self.agent is None:
            return None

        self.predict_success_count += 1

        return self.agent.predict(predict_data)

    def exploit(self, predict_data):
        """
        调用业务的exploit函数
        """
        if self.agent is None:
            return None

        self.predict_success_count += 1

        return self.agent.exploit(predict_data)

    def get_predict_stat(self):
        """
        统计预测次数
        """
        return self.predict_success_count
