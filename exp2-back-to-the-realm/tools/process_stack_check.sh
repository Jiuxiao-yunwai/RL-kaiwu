#!/bin/bash
# 查看运行中的python进程的堆栈


chmod +x tools/common.sh
. tools/common.sh

# 采用下面方法进入到需要查看堆栈的pid
/usr/local/python-3.7/bin/pyrasite-shell <pid>

python3 -c "import traceback; import sys; [print(f'Thread ID: {thread_id}\n', ''.join(traceback.format_stack(frame))) for thread_id, frame in sys._current_frames().items()]"