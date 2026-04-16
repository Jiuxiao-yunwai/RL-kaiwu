#!/bin/bash

# 一键切换进程需要配置的device, 主要确保在CPU, NPU, GPU切换


chmod +x tools/common.sh
. tools/common.sh


helper_content()
{
    echo -e "\033[31m useage: sh tools/change_process_device.sh process_name device, sh tools/change_process_device.sh learner GPU  \033[0m"
}

if [ $# -ne 2 ];
then
    helper_content

    exit -1
fi

process_name=$1
device=$2

case "$process_name" in
"aisrv"|"actor"|"learner")
    # 如果 $process_name 是 "aisrv"，"actor" 或 "learner"，执行这里的命令
    echo "The process name is $process_name"
    ;;
*)
    # 如果 $process_name 不在上述列表中，执行这里的命令
    helper_content

    exit -1
    ;;
esac

case "$device" in
"CPU"|"GPU"|"NPU")
    # 如果 $process_name 是 "aisrv"，"actor" 或 "learner"，执行这里的命令
    echo "The device is $device"
    ;;
*)
    # 如果 $process_name 不在上述列表中，执行这里的命令
    helper_content

    exit -1
    ;;
esac

# 下面是具体的修改配置文件的操作
configue_file="conf/kaiwudrl/configure.toml"
sed -i 's/${process_name}_device_type = .*/${process_name}_device_type = "'"$device"'"/' $configue_file
judge_succ_or_fail $? "change $configue_file $process_name $device"
