#!/bin/bash

# 一键切换普罗米修斯配置信息


chmod +x tools/common.sh
. tools/common.sh


if [ $# -ne 3 ];
then
    echo -e "\033[31m useage: sh tools/change_prometheus.sh prometheus_user prometheus_pwd prometheus_pushgateway \033[0m"

    exit -1
fi

prometheus_user=$1
prometheus_pwd=$2
prometheus_pushgateway=$3

# 下面是具体的修改配置文件的操作

prometheus_config_file="conf/kaiwudrl/configure.toml"
sed -i 's/prometheus_user = .*/prometheus_user = "'"$prometheus_user"'"/' $prometheus_config_file
prometheus_pwd_escaped=$(printf '%s\n' "$prometheus_pwd" | sed 's:[\\/&]:\\&:g;$!s/$/\\/')
sed -i 's/^prometheus_pwd = .*/prometheus_pwd = "'"$prometheus_pwd_escaped"'"/' $prometheus_config_file
sed -i 's/^prometheus_pushgateway = .*/prometheus_pushgateway = "'"$prometheus_pushgateway"'"/' $prometheus_config_file
judge_succ_or_fail $? "change prometheus $prometheus_user $prometheus_pwd $prometheus_pushgateway"
