#!/bin/bash

# 一键设置预加载配置的功能

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 2 ];
then
    echo -e "\033[31m usage: sh tools/preload_model_change.sh preload_model_dir preload_model_id such as: sh tools/preload_model_change.sh /data/projects/kaiwu-fwk/ckpt/ 0 \033[0m"
    exit -1
fi

configure_file=conf/configure_app.toml
preload_model_dir=$1
preload_model_id=$2

sed -i 's/preload_model = .*/preload_model = true/g' $configure_file
sed -i "s|preload_model_dir = .*|preload_model_dir = \"$preload_model_dir\"|g" $configure_file
sed -i "s/preload_model_id = .*/preload_model_id = $preload_model_id/g" $configure_file

judge_succ_or_fail $? "$configure_file change preload_model_dir $preload_model_dir preload_model_id $preload_model_id"
