#!/bin/bash
# 查看运行中进程的启动时间


chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/process_start_time.sh pid, such as sh tools/process_start_time.sh 1 \033[0m"

    exit -1
fi

pid=$1

ps -eo pid,lstart,etime | grep $pid