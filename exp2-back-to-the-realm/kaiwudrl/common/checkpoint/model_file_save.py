#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file model_file_save.py
# @brief
# @author kaiwu
# @date 2023-11-28


import os
from datetime import datetime, timezone
import re
import shutil
import tempfile
import multiprocessing
import time
import glob
import json
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from distutils.dir_util import remove_tree
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.cos_utils import COSSave
from kaiwudrl.common.checkpoint.model_path_manager import MODEL_PATH_MANGER
from kaiwudrl.common.utils.common_func import (
    get_first_last_line_from_file,
    get_sort_file_list,
    make_tar_file,
    tar_file_extract,
    make_single_dir,
    write_json_to_file,
    clean_dir,
    python_exec_shell,
    load_private_key_by_data,
    generate_private_signature_by_data,
    base64_encode,
    compute_directory_hash,
    get_map_content,
    register_sigterm_handler,
)
from kaiwudrl.common.checkpoint.model_file_common import (
    get_checkpoint_id_by_re,
)
from kaiwudrl.common.monitor.monitor_proxy import MonitorProxy


class ModelFileSave(multiprocessing.Process):
    """
    单个Model文件较大, 传递到COS耗时较多, 故采用单个进程处理
    """

    def __init__(
        self,
    ) -> None:
        super(ModelFileSave, self).__init__()

        self.exit_flag = multiprocessing.Value("b", False)

        # 只有主learner才执行上传model file任务
        self.is_checf = False

        self.local_and_remote_dirs = MODEL_PATH_MANGER.get_local_and_remote_dirs()

        # 记录进程启动时间
        self.process_start_time = time.monotonic()

        # 统计值
        self.push_to_cos_succ_cnt = 0
        self.push_to_cos_err_cnt = 0

        # 上次保存旁路的时间
        self.last_bypass_time = 0

        if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:
            # 记录当前所有的文件列表, 一般不会太大
            self.model_file_list = []
            make_single_dir(CONFIG.standard_upload_file_dir)

    def stop(self):
        self.exit_flag.value = True
        self.join()

        self.logger.info("model_file_save ModelFileSave stop success", g_not_server_label)

    def get_newest_step(self):
        """
        解析model文件下的step
        """

        model_path = f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}"

        checkpoint_file = f"{model_path}/{KaiwuDRLDefine.CHECK_POINT_FILE}"

        last_line = None
        try:
            _, last_line = get_first_last_line_from_file(checkpoint_file)
        except Exception as e:
            pass
        if not last_line or (KaiwuDRLDefine.KAIWU_MODEL_CKPT not in last_line):
            return -1

        # 格式形如all_model_checkpoint_paths: "/data/ckpt//sgame_ppo/model.ckpt-4841", 注意不要采用正则匹配, 因为app可能会有具体的数字
        checkpoint_id = re.search(r"(?<={}-)\d+".format(KaiwuDRLDefine.KAIWU_MODEL_CKPT), last_line)
        if not checkpoint_id:
            return -1
        checkpoint_id = int(checkpoint_id.group())

        return checkpoint_id

    def make_model_tar_file(self, key, local_and_remote_dir):
        """
        流程如下:
        1. 根据形如/data/ckpt/hero_ppo/checkpoint找到最新的step生成的checkpoint文件, 形如:
        all_model_checkpoint_paths: "/data/ckpt//hero_ppo/model.ckpt-3247"
        2. 在/data/ckpt/hero_ppo/ | grep 3247找出满足需求的model-3247.data-00000-of-00001,
            model-3247.index, model-3247.meta, checkpoint
        3. 对2制作成tar文件, 生成tar文件路径
        """

        if key == KaiwuDRLDefine.CKPT_DIR:
            # 放在cos_local_target_dir目录下生成tar文件
            time_str = datetime.now().strftime("%Y-%m-%d-%H-%M")

            # 将时间转换为RFC3339格式
            now = datetime.now(timezone.utc)
            created_at = now.isoformat()
            step = self.get_newest_step()

            output_file_name = (
                f"{CONFIG.cos_local_target_dir}/{CONFIG.app}_{CONFIG.algo}_{time_str}.{KaiwuDRLDefine.TAR_GZ}"
            )
            make_tar_file(output_file_name, local_and_remote_dir)

            write_json_to_file(
                {
                    "created_at": created_at,
                    "train_time": int(time.monotonic() - self.process_start_time),
                    "train_step": int(step),
                },
                f"{CONFIG.app}_{CONFIG.algo}_{time_str}",
                CONFIG.cos_local_target_dir,
            )

            # 删除/tmp的临时文件
            remove_tree(local_and_remote_dir)

            return output_file_name

        elif key == KaiwuDRLDefine.RESTORE_DIR:
            pass

        elif key == KaiwuDRLDefine.SUMMARY_DIR:
            pass

        elif key == KaiwuDRLDefine.PB_MODEL_DIR:
            pass

        else:
            pass

    def clear_dir(self):
        """
        按照需要清空文件夹下的文件
        保留最近N个文件, 规避磁盘空间占用, 原则如下:
        1. 如果是KaiwuDRL负责发送到COS文件, 则每次发送完成后会删除本地文件
        2. 如果是KaiwuDRL负责生成COS文件, 但是不会发送, 则每次开始前需要确保只有最近N个文件存在
        """
        if int(CONFIG.push_to_cos):
            return

        dir_list = get_sort_file_list(CONFIG.cos_local_target_dir, True)

        # 如果目前长度小于需要保留的长度则返回
        if len(dir_list) < CONFIG.cos_local_keep_file_num:
            return

        # 对于大于保留长度的文件进行删除操作
        for file in dir_list[CONFIG.cos_local_keep_file_num :]:
            file_path = os.path.join(CONFIG.cos_local_target_dir, file)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)

    def push_file_to_cos(self, temp_remote_dirs):
        """
        操作步骤如下:
        1. 将需要上传COS的文件目录拷贝到tmp目录下
        2. 对该目录压缩成tar.gz包
        3. 将tar.gz包上传到COS
        4. 将tmp目录下的文件目录删除
        """

        # 清空文件夹内容
        self.clear_dir()

        if not temp_remote_dirs:
            return

        for key, local_and_remote_dir in temp_remote_dirs.items():
            if MODEL_PATH_MANGER.need_to_sync(key):
                self.logger.info(
                    f"local_and_remote_dir {key}/{local_and_remote_dir} need to sync to COS",
                    g_not_server_label,
                )

                try:

                    # 注意约定的格式, 在拉取COS文件时需要设置下
                    output_file_name = self.make_model_tar_file(key, local_and_remote_dir)
                    if not output_file_name:
                        continue

                    # 如果是需要旁路, 按照旁路时间间隔旁路一份
                    if CONFIG.use_bypass:
                        # time.time()返回的是以秒为单位
                        now = int(time.time())
                        if now - self.last_bypass_time >= int(CONFIG.bypass_per_minutes) * 60:
                            if os.path.exists(output_file_name):
                                shutil.copy(output_file_name, CONFIG.bypass_dir)
                                self.last_bypass_time = now

                    """
                    由于在集群部署环境时, COS的信息无法暴露给普通使用者, 故采用的方案:
                    1. 对于有外网使用的用户, KaiwuDRL屏蔽conf下的文件, 打包到指定文件夹, 由集群开启新的容器负责传输打包的文件到COS
                    2. 对于内部用户, KaiwuDRL暴露conf下的文件, 打包, 上传到COS

                    注意如果是KaiwuDRL上传, 会删除output_file_name文件; 否则需要外界脚本来删除掉output_file_name文件
                    """
                    if int(CONFIG.push_to_cos):
                        cos_bucket_key = f"{KaiwuDRLDefine.COS_BUCKET_KEY}{CONFIG.app}"

                        key = f'{cos_bucket_key}{output_file_name.split("/")[-1]}'

                        if self.model_file_saver.push_to_cos(output_file_name, CONFIG.cos_bucket, key):
                            self.logger.info(
                                f"model_file_save file push to cos success, local_and_remote_dir {output_file_name}",
                                g_not_server_label,
                            )
                            self.push_to_cos_succ_cnt += 1
                        else:
                            self.logger.error(
                                f"model_file_save file push to cos error, local_and_remote_dir {output_file_name}",
                                g_not_server_label,
                            )
                            self.push_to_cos_err_cnt += 1

                        # 删除output_file_name
                        os.remove(output_file_name)

                except Exception as e:
                    self.logger.exception(
                        f"model_file_save push to cos error, as error is {str(e)}, ",
                        g_not_server_label,
                    )

    def before_run(self):

        # 日志处理
        self.logger = KaiwuLogger()
        pid = os.getpid()
        self.logger.setLoggerFormat(
            f"/{CONFIG.svr_name}/model_file_save_pid{pid}_log_{datetime.now().strftime('%Y-%m-%d-%H')}.log",
            "model_file_save",
        )

        # COS保存句柄, 从conf/configure下获取COS的配置
        if int(CONFIG.push_to_cos):
            self.model_file_saver = COSSave(
                self.logger,
                CONFIG.cos_secret_id,
                CONFIG.cos_secret_key,
                CONFIG.cos_region,
                CONFIG.cos_token,
            )
            self.model_file_saver.connect_to_cos()

        # 注册定时器任务
        # set_schedule_event(CONFIG.model_file_save_per_minutes, self.save_model_file_to_cos)
        # set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.model_file_save_stat)

        t = time.time()
        self.last_run_schedule_time_by_to_cos = t
        self.last_run_schedule_time_by_save_stat = t

        # 访问普罗米修斯的类
        if int(CONFIG.use_prometheus):
            self.monitor_proxy = MonitorProxy(self.logger)
            self.monitor_proxy.start()

        self.process_run_count = 0

        # 建立必要的文件目录
        make_single_dir(CONFIG.cos_local_target_dir)
        # 数字签名的私钥对象
        self.private_key = load_private_key_by_data(CONFIG.private_key_content)

        # 注册SIGTERM信号处理
        register_sigterm_handler(self.handle_sigterm, CONFIG.sigterm_pids_file)

        self.logger.info(
            f"model_file_save process start success at pid {os.getpid()}",
            g_not_server_label,
        )

        return True

    def model_file_save_stat(self):
        if CONFIG.use_prometheus:
            monitor_data = {
                KaiwuDRLDefine.PUSH_TO_COS_SUCC_CNT: self.push_to_cos_succ_cnt,
                KaiwuDRLDefine.PUSH_TO_COS_ERR_CNT: self.push_to_cos_err_cnt,
            }

            self.monitor_proxy.put_data(monitor_data)

            # 由于是一直朝上增长的指标, 不需要指标复原, 能看见代码在正常运行, 可以根据周期间隔计算出时间段内的执行次数

    def save_model_file_to_cos_normal(self):
        # 拷贝到临时目录
        temp_remote_dirs = self.copy_to_temp_dir()

        # 上传Model文件到COS
        self.push_file_to_cos(temp_remote_dirs)

        # 删除临时目录
        self.remove_temp_dir(temp_remote_dirs)

        self.logger.info(
            f"model_file_save train model file save to cos success, "
            f"succ cnt {self.push_to_cos_succ_cnt}, err cnt is {self.push_to_cos_err_cnt}",
            g_not_server_label,
        )

    def find_multiple_files(self, directory):
        """
        查找多种文件格式的情况
        """
        tar_gz_files = glob.glob(f"{directory}/*.tar.gz")
        npy_files = glob.glob(f"{directory}/*.npy")
        pkl_files = glob.glob(f"{directory}/*.pkl")
        meta_files = glob.glob(f"{directory}/*.meta")
        index_files = glob.glob(f"{directory}/*.index")

        return tar_gz_files + npy_files + pkl_files + meta_files + index_files

    def single_replace_eval_model_dir(self):

        # eval_model_dir配置文件设置为固定的
        eval_model_dir = f"/data/projects/{CONFIG.app}/ckpt"

        # 获取ID
        id_list_file = f"{CONFIG.cos_local_target_dir}/ckpt/{KaiwuDRLDefine.KAIWU_MODEL_ID_LIST}"
        # 解析ID内容
        last_line = None
        try:
            first_line, last_line = get_first_last_line_from_file(id_list_file)
        except Exception as e:
            pass

        checkpoint_id = get_checkpoint_id_by_re(last_line)
        if checkpoint_id is None:
            return False, checkpoint_id

        """
        修改文件, 采用shell比较快, 主要包括:
        1. mv文件
        2. 配置eval_model_dir和eval_model_id
        """
        configure_back_file = f"{CONFIG.cos_local_target_dir}/conf/configure_app.toml.bk"
        configure_file = f"{CONFIG.cos_local_target_dir}/conf/configure_app.toml"
        shell_query = f"cp {configure_back_file} {configure_file}"
        python_exec_shell(shell_query)

        shell_query = f'sed -i "/eval_model_dir/d" {configure_file} && sed -i "/eval_model_id/d" {configure_file}'
        python_exec_shell(shell_query)

        # python写文件内容
        with open(configure_file, "a+") as f:
            # 移动文件指针到文件的开始
            f.seek(0, os.SEEK_SET)
            # 读取所有行
            lines = f.readlines()
            # 如果文件不为空，并且最后一行不以换行符结尾
            if lines and not lines[-1].endswith("\n"):
                # 添加一个新的空行
                f.write("\n")

            f.write(f'eval_model_dir = "{eval_model_dir}"\n')
            f.write(f'eval_model_id = "{checkpoint_id}"\n')

        return True, checkpoint_id

    def multi_replace_eval_model_dir(self):
        """
        修改在进行评估时的实际路径
        """
        eval_model_dir = None

        # 获取到具体的引擎文件地址, 并且针对性的修改/conf/configure_app.toml里的eval_model_dir路径
        ckpt_file_list = self.find_multiple_files(f"{CONFIG.cos_local_target_dir}/ckpt")
        if ckpt_file_list:
            if len(ckpt_file_list) > 1:
                self.logger.error(
                    f"model_file_save multi_replace_eval_model_dir failed, "
                    f"ckpt files in {CONFIG.cos_local_target_dir}/ckpt 大于1个",
                    g_not_server_label,
                )
                return False

            if (
                KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX == CONFIG.use_which_deep_learning_framework
                or KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE == CONFIG.use_which_deep_learning_framework
                or KaiwuDRLDefine.MODEL_TENSORRT == CONFIG.use_which_deep_learning_framework
            ):
                # tensorflow的指定到文件目录就行
                eval_model_dir = f"/data/projects/{CONFIG.app}/ckpt"

            elif (
                KaiwuDRLDefine.MODEL_PYTORCH == CONFIG.use_which_deep_learning_framework
                or KaiwuDRLDefine.MODEL_NO_DEEP_LEARNING_FRAMEWORK == CONFIG.use_which_deep_learning_framework
            ):
                # pytorch指定到具体的文件
                eval_model_dir = f"/data/projects/{CONFIG.app}/ckpt/{ckpt_file_list[0].split('/')[-1]}"

            elif KaiwuDRLDefine.MODEL_TCNN == CONFIG.use_which_deep_learning_framework:
                pass

            else:
                pass

            # 修改文件, 采用shell比较快, 删除旧的后再新增
            configure_file = f"{CONFIG.cos_local_target_dir}/conf/configure_app.toml"
            shell_query = f'sed -i "/eval_model_dir/d" {configure_file} && echo "" >> {configure_file}'
            python_exec_shell(shell_query)

            # python写文件内容
            with open(configure_file, "a") as f:
                f.write(f'eval_model_dir = "{eval_model_dir}"\n')

            return True

        else:
            self.logger.error(
                f"model_file_save multi_replace_eval_model_dir failed, "
                f"can not find any ckpt files in {CONFIG.cos_local_target_dir}/ckpt",
                g_not_server_label,
            )
            return False

    def process_model_file(self, model_file):
        """
        下面是具体的操作逻辑
        1. 查看/data/user_ckpt_dir/下是否有新的tar.gz文件生成, 如果没有则返回
        2. 如果有则拷贝该文件到目录/data/cos_local_target_dir/下并且解压缩到ckpt, 注意删除对应的文件
        3. 拷贝配置项里需要拷贝的文件夹到/data/cos_local_target_dir/下, 比如conf等目录
        4. 修改configure_app.toml里关于评估的配置项
        5. 打包成xxx.zip
        6. 制作xxx.zip.json
        7. 拷贝4和5到/workspace/train/backup_model/目录下
        8. 清空/data/cos_local_target_dir/
        """
        if not model_file:
            return False

        # 多个目录按照,号分割
        copy_dirs = []
        if CONFIG.copy_dir:
            copy_dirs = CONFIG.copy_dir.split(",")

        # 如果没有处理则开始处理, model_file名字形如/data/user_ckpt_dir/gorge_walk_td_sarsa_2024-03-12-14-12-37.tar.gz
        if model_file not in self.model_file_list:

            # 清空目录, 删除上次遗留的文件
            clean_dir(CONFIG.cos_local_target_dir)

            # 拷贝tar.gz文件
            if os.path.exists(model_file):
                shutil.copy(model_file, CONFIG.cos_local_target_dir)

            # 拷贝需要的文件目录, 即CONFIG.copy_dir指定的目录, 由于比较复杂采用shell操作直接
            target_folder = CONFIG.cos_local_target_dir
            for copy_dir in copy_dirs:
                python_exec_shell(f"cp -r {copy_dir} {target_folder}/")

            # 去掉目录的最后的文件名字, 形如gorge_walk_td_sarsa_2024-03-12-14-12-37.tar.gz
            model_file_name = model_file.split("/")[-1]
            model_file_name_without_tag_gz = model_file_name.split(".tar.gz")[0]

            # 解压缩ckpt文件
            tar_file_extract(
                f"{CONFIG.cos_local_target_dir}/{model_file_name}",
                f"{CONFIG.cos_local_target_dir}/ckpt",
            )

            # 拷贝到ckpt文件夹, 注意是多了一层目录, 采用shell比较方便
            python_exec_shell(f"cd {CONFIG.cos_local_target_dir}/ckpt && cd */ && mv * ../ && cd ../ && rm -rf */")

            # 获取到具体的引擎文件地址, 并且针对性的修改/conf/configure_app.toml里的eval_model_dir路径
            ret, checkpoint_id = self.single_replace_eval_model_dir()
            if not ret:
                self.logger.error(
                    f"model_file_save single_replace_eval_model_dir failed, please check",
                    g_not_server_label,
                )

                return False

            # 拷贝json文件, 形如/data/user_ckpt_dir/back_to_the_realm-dqn-84400-2024_06_26_08_52_10-1.1.1.tar.gz
            json_file = (
                f"{os.path.dirname(model_file)}/"
                f"{KaiwuDRLDefine.KAIWUDRL_MODEL_FILE_JSON_FILE_NAME}_"
                f"{checkpoint_id}.json"
            )
            if os.path.exists(json_file):
                shutil.copy(json_file, CONFIG.cos_local_target_dir)

                # 由于按照step来命名文件的, 故文件会很多, 需要主动删除
                os.remove(json_file)

            # 删除tar.gz文件
            os.remove(f"{CONFIG.cos_local_target_dir}/{model_file_name}")

            cos_local_target_dir_kaiwu_json = f"{CONFIG.cos_local_target_dir}/{os.path.basename(json_file)}"

            # 针对json文件进行数字签名, 并且写回到新的json文件里, 原来json文件删除
            if not os.path.exists(cos_local_target_dir_kaiwu_json):
                self.logger.error(
                    f"model_file_save cos_local_target_dir_kaiwu_json {cos_local_target_dir_kaiwu_json} "
                    f"not found, please check",
                    g_not_server_label,
                )
                return False

            with open(
                cos_local_target_dir_kaiwu_json,
                "r",
            ) as f:
                data = json.load(f)
            os.remove(cos_local_target_dir_kaiwu_json)

            # 增加zip文件名字是为了对账
            data["model_file_name"] = f"{model_file_name_without_tag_gz}.zip"

            # 计算模型文件的hash值
            ckpt_dir_hash, model_files = compute_directory_hash(f"{CONFIG.cos_local_target_dir}/ckpt")
            data["model_file_hash"] = ckpt_dir_hash

            # 写入model_file_path, 注意多个文件采用数组形式, 需要加上相对路径
            model_files = ["ckpt/" + filename for filename in model_files]
            data["model_file_path"] = model_files

            output = get_map_content(data)
            private_signature = generate_private_signature_by_data(output, self.private_key)
            data["signature"] = base64_encode(private_signature)

            # zip文件名字和json文件名字
            zip_file_name = f"{CONFIG.standard_upload_file_dir}/{model_file_name_without_tag_gz}.zip"
            zip_json_file_name = f"{CONFIG.standard_upload_file_dir}/{model_file_name_without_tag_gz}.zip.json"

            # 注意是把kaiwu.json写入到ckpt文件夹下的, 此时没有带上训练step
            cos_local_target_dir_ckpt_kaiwu_json = (
                f"{CONFIG.cos_local_target_dir}/ckpt/{KaiwuDRLDefine.KAIWUDRL_MODEL_FILE_JSON_FILE_NAME}.json"
            )

            with open(
                cos_local_target_dir_ckpt_kaiwu_json,
                "w",
            ) as f:
                json.dump(data, f)

            # 整个打包成zip, 排除掉不需要打包的文件内容
            folder_path = f"conf/kaiwudrl"
            python_exec_shell(
                f"cd {CONFIG.cos_local_target_dir} && zip -r -q {zip_file_name} . -x '{folder_path}/*' -x '{folder_path}/'"
            )

            # 注意这里是需要先落zip文件再落地json文件的, 故对同一份data文件分2次写入
            with open(zip_json_file_name, "w") as f:
                json.dump(data, f)

                # 由于按照step来命名文件的, 故文件会很多, 需要主动删除
                os.remove(cos_local_target_dir_ckpt_kaiwu_json)

            self.logger.info(
                f"model_file_save zip_file_name {zip_file_name}, "
                f"zip_json_file_name {zip_json_file_name} copy to {CONFIG.standard_upload_file_dir} success",
                g_not_server_label,
            )

            # 该文件处理完成放入到列表里
            self.model_file_list.append(model_file)

            return True

    def save_model_file_to_cos_standard(self):
        """
        保存用户的模型文件, 采用的方法是读取learner进程落地的id_list文件, 而不是采用搜索tar.gz文件, 因为该方法会出现进程之间的同步问题
        1. 采用id_list文件, with open(id_list_file, 'r') as file:
        2. 采用glob的搜索方法, model_file_list = glob.glob(f"{CONFIG.user_ckpt_dir}/*.tar.gz")

        """
        try:
            # 读取id_list文件, 可能存在还没有建立的情况, 故需要判断是否存在
            id_list_file = f"{CONFIG.user_ckpt_dir}/{KaiwuDRLDefine.KAIWU_MODEL_ID_LIST}"
            if not os.path.exists(id_list_file):
                return

            with open(id_list_file, "r") as file:
                model_file_list = file.readlines()
            if not model_file_list:
                return

            for model_file in model_file_list:
                if self.process_model_file(model_file.strip()):

                    # 统计值增加
                    self.push_to_cos_succ_cnt += 1

                else:
                    # 统计值增加
                    self.push_to_cos_err_cnt += 1

        except Exception as e:
            self.logger.exception(
                f"model_file_save save_model_file_to_cos_standard failed, ",
                g_not_server_label,
            )
            # 统计值增加
            self.push_to_cos_err_cnt += 1

    # learner --> COS的model文件同步, 每隔多少分钟执行
    def save_model_file_to_cos(self):
        if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_NORMAL:
            self.save_model_file_to_cos_normal()
        elif CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:
            self.save_model_file_to_cos_standard()
        else:
            pass

    def run_once(self):

        # 启动定时器操作, 定时器里执行具体的保存操作, 但是有时间出现不执行的情况
        # schedule.run_pending()

        now = time.time()
        if now - self.last_run_schedule_time_by_to_cos > int(CONFIG.model_file_save_per_minutes) * 60:
            self.save_model_file_to_cos()
            self.last_run_schedule_time_by_to_cos = now
        elif now - self.last_run_schedule_time_by_save_stat > int(CONFIG.prometheus_stat_per_minutes) * 60:
            self.model_file_save_stat()
            self.last_run_schedule_time_by_save_stat = now
        else:
            pass

    def run(self) -> None:
        if not self.before_run():
            self.logger.error(f"model_file_save before_run failed, so return", g_not_server_label)
            return

        while not self.exit_flag.value:
            try:
                self.run_once()

                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                self.process_run_count += 1
                if self.process_run_count % CONFIG.idle_sleep_count == 0:
                    time.sleep(CONFIG.idle_sleep_second)

                    # process_run_count置0, 规避溢出
                    self.process_run_count = 0

            except Exception as e:
                self.logger.exception(
                    f"model_file_save run_once, err: {str(e)}",
                    g_not_server_label,
                )

    def remove_temp_dir(self, temp_remote_dirs):
        # 删除/tmp下临时目录
        for temp_dir in temp_remote_dirs.values():
            if os.path.exists(temp_dir):
                remove_tree(temp_dir)

    def copy_need_model_file(self, key, target_dir):
        """
        按照类型来做极小拷贝:
        KaiwuDRLDefine.CKPT_DIR : f'{self.ckpt_dir}/{CONFIG.app}_{CONFIG.algo}',
        KaiwuDRLDefine.RESTORE_DIR : self.restore_dir,
        KaiwuDRLDefine.SUMMARY_DIR : f'{self.summary_dir}/{CONFIG.app}_{CONFIG.algo}',
        KaiwuDRLDefine.PB_MODEL_DIR : self.pb_model_dir
        """
        if not target_dir or not key:
            return

        if key == KaiwuDRLDefine.CKPT_DIR:
            # 获取到Model文件路径所在的路径

            model_path = f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}"
            checkpoint_file = f"{model_path}/{KaiwuDRLDefine.CHECK_POINT_FILE}"

            checkpoint_id = self.get_newest_step()

            # 寻找包含checkpoint_id的meta, data, index
            for root, dirs, file_list in os.walk(model_path):
                # 排除指定目录
                dirs[:] = [d for d in dirs if d not in MODEL_PATH_MANGER.exclude_directories()]
                for file_name in file_list:
                    if f"{KaiwuDRLDefine.KAIWU_MODEL_CKPT}-{checkpoint_id}" in file_name:
                        if os.path.exists(os.path.join(root, file_name)):
                            shutil.copy(os.path.join(root, file_name), target_dir)

            # 需要增加checkpoint文件
            if os.path.exists(checkpoint_file):
                shutil.copy(checkpoint_file, target_dir)

        elif key == KaiwuDRLDefine.RESTORE_DIR:
            pass

        elif key == KaiwuDRLDefine.SUMMARY_DIR:
            pass

        elif key == KaiwuDRLDefine.PB_MODEL_DIR:
            pass

        else:
            pass

    def copy_to_temp_dir(self):
        # 格式形如: ckpt_dir --> /tmp/xxx
        temp_remote_dirs = {}
        # 生成/tmp下临时目录, 需要回滚时操作
        target_dir = tempfile.mkdtemp()

        try:
            for key, local_and_remote_dir in self.local_and_remote_dirs.items():

                if not os.path.exists(local_and_remote_dir):
                    continue

                # 可能会异常, 此时会跳转到Exception处理

                """
                因为每次拷贝的数据文件比较多, 故这里是先过滤, 再进行拷贝, 减少数据文件拷贝耗时
                """
                self.copy_need_model_file(key, target_dir)

                # copy_tree(local_and_remote_dir, target_dir)
                temp_remote_dirs[key] = target_dir

        except Exception as e:
            self.logger.exception(
                f"model_file_save Error copying local folders, ",
                f"err: {str(e)},",
                g_not_server_label,
            )

            # 拷贝失败时，本次停止上传COS, 删除目录
            for temp_dir in temp_remote_dirs.values():
                remove_tree(temp_dir)
            temp_remote_dirs.clear()

            # 本次异常生成的/tmp临时目录也需要删除
            remove_tree(target_dir)

        return temp_remote_dirs

    def start_actor_process_by_type(self, logger):
        """
        根据不同的启动方式进行处理:
        1. 正常启动, 无需做任何操作, tensorflow会加载容器里的空的model文件启动
        2. 加载配置文件启动, 需要从COS拉取model文件再启动, tensorflow会加载容器里的model文件启动

        因为该函数是被actor和learner调用, 故需要定义下COSSave, 注意不要和model_file_save的混淆了(进程里调用)
        """
        if not CONFIG.start_actor_learner_process_type:
            logger.info(
                f"model_file_save process start, type is {CONFIG.start_actor_learner_process_type}, "
                f"no need get mode file from cos"
            )
            return

        # COS保存句柄, 从conf/configure下获取COS的配置
        cos_saver = COSSave(
            logger,
            CONFIG.cos_secret_id,
            CONFIG.cos_secret_key,
            CONFIG.cos_region,
            CONFIG.cos_token,
        )
        cos_saver.connect_to_cos()

        # 获取当天上传最近的一次COS文件列表, 即最大可能恢复
        cos_bucket_key = f"{KaiwuDRLDefine.COS_BUCKET_KEY}{CONFIG.app}"

        file_list = cos_saver.query_object_list(CONFIG.cos_bucket, cos_bucket_key)
        if not file_list:
            logger.error(f"get cos object list is None")
            return

        file_list_content = file_list.get("Contents", None)
        if not file_list_content:
            logger.error(f"get cos object list content is None")
            return

        # 按照list[-1]去获取最新的文件
        key = file_list_content[-1].get("Key", None)
        if not key:
            logger.error(f"get cos object list content last Key is None")
            return

        destination_file_name = f"{CONFIG.ckpt_dir}{CONFIG.app}_{CONFIG.algo}/{KaiwuDRLDefine.COS_LAST_MODEL_FILE}"

        if cos_saver.get_from_cos(CONFIG.cos_bucket, key, destination_file_name):
            logger.info(f"get cos last Key success")
        else:
            logger.error(f"get cos last Key error")
            return

        # 解压tar文件到ckpt目录
        tar_file_extract(destination_file_name, f"{CONFIG.ckpt_dir}{CONFIG.app}_{CONFIG.algo}")

        # 删除临时的文件
        os.remove(destination_file_name)

        logger.info(
            f"{CONFIG.svr_name} get cos file is succ, last file is {destination_file_name}, "
            f"destination dir is {CONFIG.ckpt_dir}{CONFIG.app}_{CONFIG.algo}"
        )

    def handle_sigterm(self, sig, frame):
        # SIGTERM信号处理，等待2s，然后每2s手动尝试上传模型，直到被强制退出
        time.sleep(2)
        self.logger.info(f"model_file_save {os.getpid()} start handle SIGTERM.")
        while True:
            try:
                self.save_model_file_to_cos()
                self.model_file_save_stat()
                time.sleep(2)

            except Exception as e:
                self.logger.exception(
                    f"model_file_save run_once, err: {str(e)}, ",
                    g_not_server_label,
                )
