#!/bin/bash

# 一键切换七彩石配置信息


chmod +x tools/common.sh
. tools/common.sh


if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/change_rainbow.sh rainbow_env_name \033[0m"

    exit -1
fi

rainbow_env_name=$1

# 下面是具体的修改配置文件的操作
config_file="conf/kaiwudrl/configure.toml"
app_config_file="conf/configure_app.toml"
sed -i 's/use_rainbow = .*/use_rainbow = true/' $config_file
sed -i 's/^rainbow_env_name = .*/rainbow_env_name = "'"$rainbow_env_name"'"/' $config_file
sed -i 's/^rainbow_env_name = .*/rainbow_env_name = "'"$rainbow_env_name"'"/' $app_config_file

judge_succ_or_fail $? "change rainbow $rainbow_env_name"
