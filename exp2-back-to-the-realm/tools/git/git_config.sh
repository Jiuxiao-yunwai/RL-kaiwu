#!/bin/bash


# git的config操作

chmod +x tools/common.sh
. tools/common.sh


git config -l

judge_succ_or_fail $? "git config"