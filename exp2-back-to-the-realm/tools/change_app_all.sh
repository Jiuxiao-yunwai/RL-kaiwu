#!/bin/bash

# 在各个场景打镜像时, 采用一键来修改配置

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/change_app_all.sh app_name, sh tools/change_app_all.sh gorge_walk  \033[0m"

    exit -1
fi

app_name=$1

# 下面是具体的修改配置文件的操作
start_bash_file="tools/start.sh"
produce_config="tools/produce_config.sh"
run_mulit_learner_by_horovodrun_config="tools/run_mulit_learner_by_horovodrun.sh"
run_mulit_learner_by_openmpirun_config="tools/run_mulit_learner_by_openmpirun.sh"
main_configure="conf/kaiwudrl/configure.toml"
app_configure="conf/configure_app.toml"
modelpool_stop_file="tools/modelpool_stop.sh"
modelpool_start_file="tools/modelpool_start.sh"
gpu_iplist_file="thirdparty/model_pool_go/config/gpu.iplist"
clear_aisrv_log_file="tools/clear_aisrv_log.sh"
clear_actor_log_file="tools/clear_actor_log.sh"
clear_learner_log_file="tools/clear_learner_log.sh"

files=("$start_bash_file" "$produce_config" "$run_mulit_learner_by_horovodrun_config" \
       "$run_mulit_learner_by_openmpirun_config" "$main_configure" "$app_configure" "$modelpool_stop_file" \
       "$modelpool_start_file" "$gpu_iplist_file" "$clear_aisrv_log_file" "$clear_actor_log_file" "$clear_learner_log_file")

for file in "${files[@]}";
do
    sed -i "s|/data/projects/kaiwu-fwk|/data/projects/${app_name}|g" "$file"
    judge_succ_or_fail $? "change $file $app_name"

done