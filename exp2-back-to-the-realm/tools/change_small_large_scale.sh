#!/bin/bash
# 一键切换小规模和大规模场景下的配置

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/change_small_large_scale.sh small|large, sh tools/change_small_large_scale.sh small  \033[0m"

    exit -1
fi


small_large_scale=$1

# 下面是具体的修改配置文件的操作
app_configure_file="conf/configure_app.toml"
if [ $small_large_scale == "small" ];
then
    sed -i 's/predict_batch_size = .*/predict_batch_size = 1/' $app_configure_file
    sed -i 's/predict_local_or_remote = .*/predict_local_or_remote = "local"/' $app_configure_file

elif [ $small_large_scale == "large" ];
then
    sed -i 's/predict_batch_size = .*/predict_batch_size = 32/' $app_configure_file
    sed -i 's/predict_local_or_remote = .*/predict_local_or_remote = "remote"/' $app_configure_file

else
    echo -e "\033[31m useage: sh tools/change_small_large_scale.sh small|large, sh tools/change_small_large_scale.sh small  \033[0m"

    exit -1

fi

judge_succ_or_fail $? "change $app_configure_file $small_large_scale"

framework_config_file="conf/kaiwudrl/configure.toml"
if [ $small_large_scale == "small" ];
then
    sed -i 's/predict_local_or_remote = .*/predict_local_or_remote = "local"/' $framework_config_file
elif [ $small_large_scale == "large" ];
then
    sed -i 's/predict_local_or_remote = .*/predict_local_or_remote = "remote"/' $framework_config_file
else
    echo -e "\033[31m useage: sh tools/change_small_large_scale.sh small|large, sh tools/change_small_large_scale.sh small  \033[0m"

    exit -1

fi
judge_succ_or_fail $? "change $framework_config_file $small_large_scale"