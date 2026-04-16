#!/bin/bash

# 机器上采用strace启动任务, 查看输出

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 2 ];
then
    echo -e "\033[31m useage: sh tools/strace_pid.sh cmd output_file \033[0m"
    exit -1
fi

cmd=$1
output_file=$2

strace -o $output_file -f $cmd
