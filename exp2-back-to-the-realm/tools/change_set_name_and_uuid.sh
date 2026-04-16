#!/bin/bash

# 一键切换set_name, uuid


chmod +x tools/common.sh
. tools/common.sh


if [ $# -ne 2 ];
then
    echo -e "\033[31m useage: sh tools/change_set_name_and_uuid.sh set_name uuid, sh tools/change_set_name_and_uuid.sh set10 uuid10  \033[0m"

    exit -1
fi

set_name=$1
uuid=$2

# 下面是具体的修改配置文件的操作
set_name_configure_file="conf/configure_app.toml"
sed -i 's/set_name = .*/set_name = "'"$set_name"'"/' $set_name_configure_file
judge_succ_or_fail $? "change $set_name_configure_file $set_name"

uuid_config_file="conf/kaiwudrl/configure.toml"
sed -i 's/task_id = .*/task_id = "'"$uuid"'"/' $uuid_config_file
judge_succ_or_fail $? "change $uuid_config_file $uuid"