#!/bin/bash


# python里查看内存泄漏的方法, 需要先安装pip3 install memray

chmod +x tools/common.sh
. tools/common.sh


# 参数如下:
# pid, 进程ID
if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/python_memory_memray.sh pid, such as sh tools/python_memory_memray.sh 1 \033[0m"

    exit -1
fi

pid=$1

# 直接attach到运行中的进程
memray attach $pid -o memray_output.bin

# attch成功生成的memray_output.bin可以用于分析, 主要是下面的操作, 需要等生产一段时间后, 再生成火焰图, 表格等
memray flamegraph memray_output.bin

memray table memray_output.bin

memray summary memray_output.bin
