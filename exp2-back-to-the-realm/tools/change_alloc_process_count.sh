#!/bin/bash
# 更新配置文件里的进程个数的配置



chmod +x tools/common.sh
. tools/common.sh


if [ $# -ne 2 ];
then
    echo -e "\033[31m useage: sh tools/change_alloc_process_count.sh aisrv|actor|learner|kaiwu_env count \
    such as: sh tools/change_alloc_process_count.sh aisrv 1  \033[0m"

    exit -1
fi

server_name=$1
count=$2

# 同时修改下面的配置文件
configure_file=conf/kaiwudrl/configure.toml
aisrv_configure_file=conf/kaiwudrl/aisrv.toml
app_configure_file=conf/configure_app.toml

if [ $server_name == "aisrv" ];
then
    echo -e "\033[32m aisrv not need to change alloc process count \033[0m"
    exit -1

elif [ $server_name == "actor" ];
then
    sed -i "s/aisrv_connect_to_actor_count = .*/aisrv_connect_to_actor_count = $count/g" $aisrv_configure_file
    sed -i "s/aisrv_connect_to_actor_count = .*/aisrv_connect_to_actor_count = $count/g" $app_configure_file

elif [ $server_name == "learner" ];
then
    sed -i "s/aisrv_connect_to_learner_count = .*/aisrv_connect_to_learner_count = $count/g" $aisrv_configure_file
    sed -i "s/aisrv_connect_to_learner_count = .*/aisrv_connect_to_learner_count = $count/g" $app_configure_file

elif [ $server_name == "kaiwu_env" ];
then
    sed -i "s/aisrv_connect_to_kaiwu_env_count = .*/aisrv_connect_to_kaiwu_env_count = $count/g" $configure_file
    sed -i "s/aisrv_connect_to_kaiwu_env_count = .*/aisrv_connect_to_kaiwu_env_count = $count/g" $app_configure_file

else
    echo -e "\033[31m useage: sh tools/change_alloc_process_count.sh aisrv|actor|learner|kaiwu_env count \
    such as: sh tools/change_alloc_process_count.sh aisrv 1  \033[0m"

    exit -1
fi

judge_succ_or_fail $? "$server_name change alloc process count $count"
