#!/bin/bash
# 查看learner机器上train日志里的训练步骤



chmod +x tools/common.sh
. tools/common.sh


# 通过查看learner生成的日志来确认训练的次数
function get_learner_train_stats()
{
    learner_train_log=log/learner/mpirun.log
    if [ ! -f "$learner_train_log" ]; 
    then
        learner_train_log=log/learner.log
    fi

    learner_train_size=$(grep -oP "tensorflow global step is \K\d+" $learner_train_log | tail -n 1)
    if [ -z "$learner_train_size" ]; 
    then
        learner_train_size=0
    fi

    echo "$learner_train_size"
}


learner_train_size=$(get_learner_train_stats)
echo "learner train size: "$learner_train_size
