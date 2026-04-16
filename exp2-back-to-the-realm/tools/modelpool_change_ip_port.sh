#!/bin/bash

# modelpool进程替换端口, 注意是2个端口, 默认的10013是modelpool进程的端口, 10014是提供http服务的端口

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 2 ];
then
    echo -e "\033[31m usage: sh tools/modelpool_change_ip_port.sh modelpool_port modelpool_http_port \033[0m"
    exit -1
fi

modelpool_port=$1
modelpool_http_port=$2

# 下面是具体的操作
sed -i "s/:10014/:$modelpool_http_port/g" tools/produce_config.sh
grep -rl --include=\*.yaml "10014" thirdparty/model_pool_go/ | xargs sed -i "s/:10014/:${modelpool_http_port}/g"
grep -rl --include=\*.yaml "10014" thirdparty/model_pool_go/ | xargs sed -i "s/port: 10014/port: ${modelpool_http_port}/g"
grep -rl "10013" thirdparty/model_pool_go/ | xargs sed -i "s/port: 10013/port: ${modelpool_port}/g"
grep -rl "10013" thirdparty/model_pool_go/ | xargs sed -i "s/:10013/:${modelpool_port}/g"
