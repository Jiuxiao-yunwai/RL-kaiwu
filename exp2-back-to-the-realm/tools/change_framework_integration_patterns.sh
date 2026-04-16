#!/bin/bash
# 一键切换框架接入模式

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/change_framework_integration_patterns.sh standard|normal, sh tools/change_framework_integration_patterns.sh normal \033[0m"

    exit -1
fi

framework_integration_patterns=$1

# 下面是具体的修改配置文件的操作
config_file="conf/kaiwudrl/configure.toml"
sed -i 's/framework_integration_patterns = .*/framework_integration_patterns = "'"$framework_integration_patterns"'"/' $config_file
judge_succ_or_fail $? "change $config_file $framework_integration_patterns"
