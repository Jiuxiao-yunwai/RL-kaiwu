#!/bin/bash


# produce_config.sh 主要生成配置文件内容
# 目的是减少人工操作
# 如果job_master上线后, 则不需要采用该脚本来生成配置, 而是依赖job_master的生成配置和部署

chmod +x tools/common.sh
. tools/common.sh

main_configure_file=conf/kaiwudrl/configure.toml
app_configure_file=conf/configure_app.toml
gpu_configure_file=thirdparty/model_pool_go/config/gpu.iplist
openmpi_congiure_file=tools/run_mulit_learner_by_openmpirun.sh

# 配置leaerner上的openmpirun的ip列表
change_openmpirun_gpu_list()
{
    learner_ip_list_str=$1

    learner_ip_list=(`echo $learner_ip_list_str | tr ',' ' '`)
    count=0
    for ip_port in ${learner_ip_list[@]}
    do
        # 字符串切割为IP和port的形式
        arr=(`echo $ip_port | tr ':' ' '`)
        len=${#arr[@]}
        if [ $len -eq 2 ];
        then
            ip=${arr[0]}
            port=${arr[1]}
        elif [ $len -eq 1 ];
        then
            ip=${arr[0]}
            port=9999
        else
            echo "$ip_port is not ip:port or ip format"
            exit -1
        fi

        if [ $count -eq 0 ];
        then
            learner_addrs_openmpi_list="\"$ip:1\""
        else
            learner_addrs_openmpi_list="$learner_addrs_openmpi_list,"\"$ip:1\"""
        fi

        let count++
    done

    # 修改以某个字符串开头的某行的配置项
    sed -i '/^Nodelist=/c'Nodelist=$learner_addrs_openmpi_list'' $openmpi_congiure_file
    sed -i '/^Num_process=/c'Num_process=$count'' $openmpi_congiure_file
}


# 配置leaner上modelpool的gpu_list列表, 参数为按照逗号分割的字符串
change_modelpool_gpu_list()
{
    learner_ip_list_str=$1

    learner_ip_list=(`echo $learner_ip_list_str | tr ',' ' '`)
    cat /dev/null > $gpu_configure_file

    for ip_port in ${learner_ip_list[@]}
    do
        # 字符串切割为IP和port的形式
        arr=(`echo $ip_port | tr ':' ' '`)
        len=${#arr[@]}
        if [ $len -eq 2 ];
        then
            ip=${arr[0]}
            port=${arr[1]}
        elif [ $len -eq 1 ];
        then
            ip=${arr[0]}
            port=9999
        else
            echo "$ip_port is not ip:port or ip format"
            exit -1
        fi

        echo $ip >> $gpu_configure_file
    done
}

# 根据进程名来生成不同的配置
# 针对leaner的操作如下:
#   根据传入的learner的IP列表, 修改run_mulit_learner_by_openmpirun.sh配置, modelpool地址
# 针对actor的操作如下:
#   根据传入的actor的IP列表, 修改modelpool地址
# 针对aisrv的操作如下:
#   根据传入的learner的IP列表, 修改modelpool地址
produce_config_by_process_name()
{
    process_name=$1
    configure_file=$2

    # 不同的进程对这些参数使用方法不同, 为了编程和运营方便, 这里参数形式和个数不按照进程名来区别
    # aisrv, 分为需要learner列表和不需要learner列表2种方式
    # actor, 需要learner列表
    # learner, 需要learner列表
    # learner_ip_list_str支持IP:端口形式的; 同时也支持IP形式的, 此时端口即采用默认端口
    learner_ip_list_str=$3

    # 获取本机IP
    get_host_ip
    host_ip=$result

    # 不同的进程名, 生成不同的配置文件
    if [ $process_name == "aisrv" ];
    then

        # 如果是采用aisrv集成了actor的predict作用, 则需要启动modelpool, 并且配置gpu_list
        predict_local_or_remote=$(grep -oP 'predict_local_or_remote\s*=\s*["'\'']\K[^"'\''"]+' $app_configure_file 2>/dev/null || true)
        if [ -z "$predict_local_or_remote" ];
        then
            predict_local_or_remote=$(grep -oP 'predict_local_or_remote\s*=\s*["'\'']\K[^"'\''"]+' $main_configure_file 2>/dev/null || true)
        fi

        if [ "$predict_local_or_remote" == "local" ];
        then

            # 修改modelpool_remote_addrs内容
            sed -i 's/modelpool_remote_addrs = .*/modelpool_remote_addrs = "'"$host_ip:10014"'"/' $configure_file

            # 配置thirdparty/model_pool_go/config下面的gpu.iplist
            change_modelpool_gpu_list $learner_ip_list_str

        fi
    elif [ $process_name == "actor" ];
    then

        # 修改modelpool_remote_addrs内容
        sed -i 's/modelpool_remote_addrs = .*/modelpool_remote_addrs = "'"$host_ip:10014"'"/' $configure_file

        # 配置thirdparty/model_pool_go/config下面的gpu.iplist
        change_modelpool_gpu_list $learner_ip_list_str

    elif [ $process_name == "learner" ];
    then
        # 修改modelpool_remote_addrs内容
        sed -i 's/modelpool_remote_addrs = .*/modelpool_remote_addrs = "'"$host_ip:10014"'"/' $configure_file
        sed -i 's/ip_address = .*/ip_address = "'"$host_ip"'"/' $configure_file

        # 配置thirdparty/model_pool_go/config下面的gpu.iplist
        change_modelpool_gpu_list $learner_ip_list_str

        # 配置tools下的run_mulit_learner_by_openmpirun.sh里的Nodelist和Num_process
        # change_openmpirun_gpu_list $learner_ip_list_str

    fi
}
