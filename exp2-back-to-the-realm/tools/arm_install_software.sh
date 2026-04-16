#!/bin/bash


# 部分软件需要在arm上重新安装, 因为其使用到的C++需要分不同的CPU架构

chmod +x tools/common.sh
. tools/common.sh


# 安装的软件列表如下, 注意这里列出来具体的版本号, 如果出现某些软件的版本已经变化了, 就需要按照实际修改
array=("zmq" "toml")
for element in ${array[@]}
do
    pip3 install $element
    judge_succ_or_fail $? "$element install"
done


judge_succ_or_fail $? "cp arm install software"
