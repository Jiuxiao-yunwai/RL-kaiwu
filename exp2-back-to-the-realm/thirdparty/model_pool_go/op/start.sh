#!/bin/bash

# 下面是modelpool部署方式:
# 1. 角色分为actor和learner上容器的进程
# 2. 主learner(其他learner不需要启动)进程, 启动sh start.sh learner
# 3. 每个actor进程, 启动sh start.sh actor

if [ $# -ne 1 ];
then
   echo "usage sh start.sh actor|learner, such as: sh start.sh learner"
   exit -1
fi

role=$1

if [ -d "../log" ];
then
    rm -r ../log
fi
mkdir ../log

# 获取操作系统架构
get_machine_architecture_information()
{
    result=$(uname -m)
    if [ -n "$result" ];
    then
        machine_architecture_information=$result
    fi
}

get_machine_architecture_information
machine_architecture_information=$result
if [ $machine_architecture_information == "x86_64" ];
then
   modelpool_file=../bin/x86/modelpool
   modelpool_proxy_file=../bin/x86/modelpool_proxy
elif [ $machine_architecture_information == "aarch64" ];
then
   modelpool_file=../bin/aarch64/modelpool
   modelpool_proxy_file=../bin/aarch64/modelpool_proxy

# 未来扩展, 暂时按照reverb
else
   modelpool_file=../bin/x86/modelpool
   modelpool_proxy_file=../bin/x86/modelpool_proxy
fi

# actor进程
if [ $role == "actor" ];
then
   # 获取master_ip, 注意是从config下的gpu.iplist获取的, 而gpu.iplist 是需要配置的, 每行一个IP
   master_ip=`head -n 1 ../config/gpu.iplist | awk '{print $1}'`
   bash set_actor_config.sh $master_ip
   nohup ./${modelpool_file} -conf=../config/trpc_go.yaml > ../log/cpu.log 2>&1 &
   nohup ./${modelpool_proxy_file} -fileSavePath=./model > ../log/proxy.log 2>&1 &
# learner进程
elif [ $role == "learner" ];
then
   bash set_learner_config.sh
   nohup ./${modelpool_file} -conf=../config/trpc_go.yaml > ../log/gpu.log 2>&1 &
   nohup ./${modelpool_proxy_file} -fileSavePath=./model > ../log/proxy.log 2>&1 &
else
   echo "usage sh start.sh actor|learner, such as: sh start.sh learner"
   exit -1
fi
