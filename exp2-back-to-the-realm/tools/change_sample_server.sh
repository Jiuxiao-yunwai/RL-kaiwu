#!/bin/bash
# 一键样本处理器的配置, 比如reverb, zmq等

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/change_sample_server.sh reverb|zmq, sh tools/change_sample_server.sh reverb  \033[0m"

    exit -1
fi


sample_server_type=$1

# 下面是具体的修改配置文件的操作
learner_configure_file="conf/kaiwudrl/learner.toml"
configure_file="conf/kaiwudrl/configure.toml"
if [ $sample_server_type == "reverb" ];
then
    sed -i 's/use_learner_server = .*/use_learner_server = false/' $learner_configure_file
    sed -i 's/replay_buffer_type = .*/replay_buffer_type = "reverb"/' $configure_file

elif [ $sample_server_type == "zmq" ];
then
    sed -i 's/use_learner_server = .*/use_learner_server = true/' $learner_configure_file
    sed -i 's/replay_buffer_type = .*/replay_buffer_type = "zmq"/' $configure_file

else
    echo -e "\033[31m useage: sh tools/change_sample_server.sh reverb|zmq, sh tools/change_sample_server.sh reverb  \033[0m"

    exit -1

fi

judge_succ_or_fail $? "change $learner_configure_file $sample_server_type"
judge_succ_or_fail $? "change $configure_file $sample_server_type"
