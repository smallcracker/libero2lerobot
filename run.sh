#!/bin/bash
cd /home/longlive/libero2lerobot
source ./.venv/bin/activate

# 确保outputs目录存在
mkdir -p outputs

# 定义log文件路径
LOG_FILE="outputs/run.log"

# 启动python命令并将输出重定向到log文件，同时在后台运行
python ./libero_rlds_converter.py --data-dir ~/stack_cube_datasets/hdf5/ --repo-id stack-cube/franka --private --use-videos --clean-existing >> "$LOG_FILE" 2>&1 &

# 获取后台进程的PID
PID=$!

# 使用tail实时显示日志
tail -f "$LOG_FILE" --pid=$PID
