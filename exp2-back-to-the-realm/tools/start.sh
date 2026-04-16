#!/bin/bash

# 进程启动脚本
# 主要是下面的进程:
# aisrv, 单独拉aisrv进程, 需要区分是否使用alloc服务
#   如果使用alloc服务, 依赖learner启动和, 如果是标准化, 则依赖kaiwu_env启动
#   如果不使用alloc服务, 不依赖learner启动, 如果是标准化, 则依赖kaiwu_env启动
# actor, 单独拉起actor进程
# leaerner, 分为下面模式
#    cluster模式, 采用openmpi等启动
#    single模式, 不用采用openmpi等启动, 而是采用python3启动即可
# modelpool, 拉起来modelpool进程
# all, 这种场景是开发测试使用, 故会启动aisrv, actor, learner single, modelpool learner

chmod +x tools/common.sh
. tools/common.sh

chmod +x tools/produce_config.sh
. tools/produce_config.sh

chmod +x tools/change_linux_parameter.sh
. tools/change_linux_parameter.sh

# 参数的场景比较多, 主要分为:
# process, 进程名字, 必选
# learner_ips, learner IP列表
# 如果是需要启动learner进程族, 必选
# 如果是不需要启动learner进程族, 可选
# game_id, 可选
if [ $# -ne 1 ] && [ $# -ne 2 ] && [ $# -ne 3 ];
then
    echo -e "\033[31m useage: sh tools/start.sh all|actor|aisrv|learner [learner_ips] [game_id] \
     \n such as: sh tools/start.sh all 127.0.0.1,127.0.0.2 \
     \n or sh tools/start.sh learner 127.0.0.1,127.0.0.2 \
     \n or sh tools/start.sh aisrv 127.0.0.1,127.0.0.2 game_id \
     \n or sh tools/start.sh aisrv \
     \033[0m"

    exit -1
fi

# 启动modelpool进程
# 参数process, 主要区分actor和learner
# 启动前, 删除前次遗留的文件
start_modelpool()
{
    process=$1

    # 删除掉上次运行后遗留的文件
    cd thirdparty/model_pool_go/bin/
    rm -rf files/*
    rm -rf model/*
    judge_succ_or_fail $? "modelpool $process old file delete"

    # 注意文件目录路径
    cd ../op/
    sh start.sh $process
    judge_succ_or_fail $? "modelpool $process start"
    cd /data/projects/back_to_the_realm
}

# 无论哪种进程启动, 先修改下linux内核参数, 注意部分机器适配的情况
change_tcp_parameter
judge_succ_or_fail $? "change_tcp_parameter"

# 删除已经有的core文件
rm -rf /data/corefile/*

# 进程启动日志, 输入到日志文件里, 规避出现问题时不知道报错信息
work_dir=/data/projects/back_to_the_realm
log_dir=${LOG_DIR:-$work_dir/log}

aisrv_start_file=$log_dir/aisrv.log
actor_start_file=$log_dir/actor.log
learner_start_file=$log_dir/learner.log
client_start_file=$log_dir/client.log

# 配置protobuf的路径
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/protobuf/lib
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/protobuf/lib
export PATH=$PATH:/usr/local/protobuf/bin

source /etc/profile
ldconfig

# 在评估时不需要启动modelpool则这里设置为按照需要启动
START_MODELPOOL="${START_MODELPOOL:-true}"

make_log_dir()
{
    if [ ! -x "$log_dir/" ];
    then
        mkdir $log_dir/
    fi
}

# 下面日志文件, 如果日志目录没有在则新建, 在则清空日志目录, 规避日志多导致机器磁盘紧张
make_aisrv_log_dir()
{
    make_log_dir
    if [ ! -x "$log_dir/aisrv" ];
    then
        mkdir $log_dir/aisrv
    else
        find $log_dir/aisrv/ -type f -exec rm -f {} +
    fi
}

make_actor_log_dir()
{
    make_log_dir
    if [ ! -x "$log_dir/actor" ];
    then
        mkdir $log_dir/actor
    else
        find $log_dir/actor/ -type f -exec rm -f {} +
    fi
}

make_learner_log_dir()
{
    make_log_dir
    if [ ! -x "$log_dir/learner" ];
    then
        mkdir $log_dir/learner
    else
        find $log_dir/learner/ -type f -exec rm -f {} +
    fi
}

make_client_log_dir()
{
    make_log_dir
    if [ ! -x "$log_dir/client" ];
    then
        mkdir $log_dir/client
    else
        find $log_dir/client/ -type f -exec rm -f {} +
    fi
}


# 处理机器架构信息, 因为arm机器上要采用zmq来发送样本
change_sample_server_by_machine_architecture_information()
{
    get_machine_architecture_information
    machine_architecture_information=$result
    echo -e "\033[32m machine architecture information: $machine_architecture_information  \033[0m"
    if [ $machine_architecture_information == "x86_64" ];
    then
        sh tools/change_sample_server.sh reverb
    elif [ $machine_architecture_information == "aarch64" ] || [ $machine_architecture_information == "arm64" ];
    then
        sh tools/change_sample_server.sh zmq

    # 未来扩展, 暂时按照reverb
    else
        sh tools/change_sample_server.sh reverb
    fi
}

# 传入参数
server_type=$1
learner_ips=$2
game_id=${GAME_ID:-$3}

# 全局配置文件
main_configure_file=conf/kaiwudrl/configure.toml
app_configure_file=conf/configure_app.toml
aisrv_configure_file=conf/kaiwudrl/aisrv.toml

if [ $server_type == "aisrv" ];
then

    # 如果传入的learner_ips为空
    #   代表不需要使用alloc服务, 需要事先配置use_alloc项
    #   代表不是采用小规模模型
    # 如果传入的learner_ips非空
    #   则代表需要使用alloc服务, 需要事先配置use_alloc项
    #   采用的是小规模模型

    # 先在业务配置文件里寻找, 如果没有就在KaiwuDRL配置文件里寻找
    predict_local_or_remote=$(grep -oP 'predict_local_or_remote\s*=\s*["'\'']\K[^"'\''"]+' $app_configure_file 2>/dev/null || true)
    if [ -z "$predict_local_or_remote" ];
    then
        predict_local_or_remote=$(grep -oP 'predict_local_or_remote\s*=\s*["'\'']\K[^"'\''"]+' $main_configure_file 2>/dev/null || true)
    fi

    if [ "$predict_local_or_remote" == "local" ];
    then
        # 判断learner_ip_list是否为空, 如果为空则只是拉起aisrv进程族, 如果非空则是需要拉起aisrv和learner进程族
        # 如果不需要开启learner进程, 但是传递了learner地址, 传递的learner地址不做使用
        # 如果需要开启learner进程, 但是没有传递learner知道, 报错
        need_to_start_learner=$(grep -oP 'need_to_start_learner\s*=\s*\K[^#\r\n]+' $aisrv_configure_file 2>/dev/null || true)
        if [ "$need_to_start_learner" == "true" ];
        then
            check_param_is_null "$learner_ips" "learner ips is null, please check"
        fi

        if [ -n "$learner_ips" ]; then
            # 配置文件修改
            produce_config_by_process_name aisrv conf/kaiwudrl/aisrv.toml $learner_ips
            judge_succ_or_fail $? "aisrv produce config"
        fi

        # 先在业务配置文件里寻找, 如果没有就在KaiwuDRL配置文件里寻找
        run_mode=$(grep -oP 'run_mode\s*=\s*["'\'']\K[^"'\''"]+' $app_configure_file 2>/dev/null || true)
        if [ -z "$run_mode" ];
        then
            run_mode=$(grep -oP 'run_mode\s*=\s*["'\'']\K[^"'\''"]+' $main_configure_file 2>/dev/null || true)
        fi

        # 如果是采用aisrv集成了actor的predict作用, 并且为train模式, 则需要启动modelpool
        if [ "$run_mode" == "train" ];
        then
            start_modelpool "actor"
        fi
    fi

    # 生成aisrv日志
    make_aisrv_log_dir

    # 为了做到进程发送数据均衡, 这里会增加随机sleep
    sleep_time=$((RANDOM%4))
    sleep $sleep_time

    # 临时增加启动进程检测脚本
    aisrv_framework=`grep '^aisrv_framework' conf/kaiwudrl/aisrv.toml | cut -d '=' -f2 | tr -d ' '`
    if [ "$aisrv_framework" = "kaiwudrl" ];
    then
        sh tools/deploy/check_process.sh &
    fi

    #  启动进程
    if [ -n "$game_id" ];
    then
        export G6SHMNAME=KaiwuDRL && python3 kaiwudrl/server/aisrv/aisrv.py --conf=/data/projects/back_to_the_realm/conf/kaiwudrl/aisrv.toml --game_id=$game_id >$aisrv_start_file 2>&1 &
    else
        export G6SHMNAME=KaiwuDRL && python3 kaiwudrl/server/aisrv/aisrv.py --conf=/data/projects/back_to_the_realm/conf/kaiwudrl/aisrv.toml >$aisrv_start_file 2>&1 &
    fi

    judge_succ_or_fail $? "aisrv start"

elif [ $server_type == "actor" ];
then
    # 判断learner_ip_list是否为空
    check_param_is_null "$learner_ips" "learner ips is null, please check"

    # 生成actor日志文件
    make_actor_log_dir

    # 生成配置文件
    produce_config_by_process_name actor conf/kaiwudrl/actor.toml $learner_ips
    judge_succ_or_fail $? "actor produce config"

    # 为了简化k8s启动流程, 会主动拉起来modelpool actor进程
    if [ "$START_MODELPOOL" = "true" ];
    then
        # 如果是true，则执行start_modelpool命令
        start_modelpool "actor"
    else
        # 如果不是true，可以在这里执行其他操作，或者什么都不做
        echo "Skipping start_modelpool command"
    fi

    # 处理GPU异构的情况
    check_gpu_machine_type
    gpu_machine_type=$result

    echo -e "\033[32m gpu engine is $gpu_machine_type \033[0m"
    if [ -n "$gpu_machine_type" ];
    then
        sh tools/actor_cpp_copy.sh $gpu_machine_type
    fi

    # 删除/dev/shm
    rm -rf /dev/shm/*

    # 启动进程
    if [ -n "$game_id" ];
    then
        export G6SHMNAME=KaiwuDRL && python3 kaiwudrl/server/actor/actor.py --conf=/data/projects/back_to_the_realm/conf/kaiwudrl/actor.toml --game_id=$game_id >$actor_start_file 2>&1 &
    else
        export G6SHMNAME=KaiwuDRL && python3 kaiwudrl/server/actor/actor.py --conf=/data/projects/back_to_the_realm/conf/kaiwudrl/actor.toml >$actor_start_file 2>&1 &
    fi
    judge_succ_or_fail $? "actor start"

elif [ $server_type == "learner" ];
then
    # 判断learner_ip_list是否为空
    check_param_is_null "$learner_ips" "learner ips is null, please check"

    # learner需要采用该配置
    change_sample_server_by_machine_architecture_information

    # 生成learner日志文件
    make_learner_log_dir

    # 生成配置文件
    produce_config_by_process_name learner conf/kaiwudrl/learner.toml $learner_ips
    judge_succ_or_fail $? "learner produce config"

    # 为了简化k8s启动流程, 会主动拉起来modelpool learner进程
    if [ "$START_MODELPOOL" = "true" ];
    then
        # 如果是true，则执行start_modelpool命令
        start_modelpool "learner"
    else
        # 如果不是true，可以在这里执行其他操作，或者什么都不做
        echo "Skipping start_modelpool command"
    fi

    # 启动进程
    # sh tools/run_mulit_learner_by_openmpirun.sh release >/dev/null 2>&1 &
    python3 kaiwudrl/server/learner/learner.py --conf=/data/projects/back_to_the_realm/conf/kaiwudrl/learner.toml >$learner_start_file 2>&1 &
    judge_succ_or_fail $? "learner cluster release start"

elif [ $server_type == "all" ];
then

    # all的场景是测试场景, 即单个容器里启动全部进程
    produce_config_by_process_name aisrv conf/kaiwudrl/aisrv.toml 127.0.0.1
    judge_succ_or_fail $? "aisrv produce config"

    produce_config_by_process_name actor conf/kaiwudrl/actor.toml 127.0.0.1
    judge_succ_or_fail $? "actor produce config"

    produce_config_by_process_name learner conf/kaiwudrl/learner.toml 127.0.0.1
    judge_succ_or_fail $? "learner produce config"

    # 依赖的第三方组件modelpool是需要独立部署的, 如果是开发测试环境可以单独手动启动, 为了简化k8s启动流程, 会主动拉起来modelpool learner进程
    start_modelpool "learner"

    # all场景下需要采用该配置
    change_sample_server_by_machine_architecture_information

    cd /data/projects/back_to_the_realm

    make_aisrv_log_dir
    make_actor_log_dir
    make_learner_log_dir

    # 处理GPU异构的情况
    check_gpu_machine_type
    gpu_machine_type=$result

    echo -e "\033[32m gpu engine is $gpu_machine_type \033[0m"
    if [ -n "$gpu_machine_type" ];
    then
        sh tools/actor_cpp_copy.sh $gpu_machine_type
    fi

    # 如果需要actor远程预测, 则需要启动的进程为aisrv、actor、learner
    # 如果需要aisrv本地预测, 则需要启动的进程为aisrv、learner
    predict_local_or_remote=$(grep -oP 'predict_local_or_remote\s*=\s*["'\'']\K[^"'\''"]+' $app_configure_file 2>/dev/null || true)
    if [ -z "$predict_local_or_remote" ];
    then
        predict_local_or_remote=$(grep -oP 'predict_local_or_remote\s*=\s*["'\'']\K[^"'\''"]+' $main_configure_file 2>/dev/null || true)
    fi

    if [ "$predict_local_or_remote" == "remote" ];
    then
        if [ -n "$game_id" ];
        then
            export G6SHMNAME=KaiwuDRL && python3 kaiwudrl/server/actor/actor.py --conf=/data/projects/back_to_the_realm/conf/kaiwudrl/actor.toml --game_id=$game_id >$actor_start_file 2>&1 &
        else
            export G6SHMNAME=KaiwuDRL && python3 kaiwudrl/server/actor/actor.py --conf=/data/projects/back_to_the_realm/conf/kaiwudrl/actor.toml >$actor_start_file 2>&1 &
        fi
        judge_succ_or_fail $? "actor start"
    fi

    # 因为单个机器上部署多个进程, 如果启动集群版本的learner可能因为没有安装mpirun而失败
    python3 kaiwudrl/server/learner/learner.py --conf=/data/projects/back_to_the_realm/conf/kaiwudrl/learner.toml >$learner_start_file 2>&1 &
    judge_succ_or_fail $? "learner single start"

    # 最后启动aisrv, 间隔15s
    sleep 15
    if [ -n "$game_id" ];
    then
        export G6SHMNAME=KaiwuDRL && python3 kaiwudrl/server/aisrv/aisrv.py --conf=/data/projects/back_to_the_realm/conf/kaiwudrl/aisrv.toml --game_id=$game_id >$aisrv_start_file 2>&1 &
    else
        export G6SHMNAME=KaiwuDRL && python3 kaiwudrl/server/aisrv/aisrv.py --conf=/data/projects/back_to_the_realm/conf/kaiwudrl/aisrv.toml >$aisrv_start_file 2>&1 &
    fi
    judge_succ_or_fail $? "aisrv start"

# 增加上sgame_client, 便于开发测试
elif [ $server_type == "client" ];
then
    sgame_type=$3
    self_play=$4

    make_client_log_dir

    if [ $sgame_type == "sgame_5v5" ];
    then
        cd /data/projects/back_to_the_realm/app/sgame_5v5/tools
    elif [ $sgame_type == "sgame_1v1" ];
    then
        cd /data/projects/back_to_the_realm/app/sgame_1v1/tools
    else
        echo -e "\033[31m sgame_type only support sgame_5v5 or sgame_1v1 \033[0m"
        exit -1
    fi

    # 启动sgame_client
    sh start_multi_game.sh 1 /data/projects/back_to_the_realm/conf/kaiwudrl/client.toml >$client_start_file 2>&1 &
    judge_succ_or_fail $? "sgame_client start"

    cd /data/projects/back_to_the_realm/

else
    echo -e "\033[31m useage: sh tools/start.sh all|actor|aisrv|learner [learner_ips] \
     \n such as: sh tools/start.sh all 127.0.0.1,127.0.0.2 \
     \n or sh tools/start.sh learner 127.0.0.1,127.0.0.2 \
     \n or sh tools/start.sh aisrv 127.0.0.1,127.0.0.2 \
     \n or sh tools/start.sh aisrv \
     \033[0m"

    exit -1
fi
