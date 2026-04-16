#!/bin/bash


# 采用black一键对代码进行格式化操作

chmod +x tools/common.sh
. tools/common.sh


# 注意增加参数格式
black --line-length 120 *
