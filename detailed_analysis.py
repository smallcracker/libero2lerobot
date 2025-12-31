#!/usr/bin/env python3
"""
详细分析Libero数据集的demo信息
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_demos(dataset_path):
    dataset_path = Path(dataset_path).expanduser()
    data_dir = dataset_path / 'data'
    parquet_files = sorted(list(data_dir.rglob('*.parquet')))

    # 收集所有episode信息
    all_episodes_info = {}

    print('详细Demo分析:')
    print('=' * 80)

    print('正在分析parquet文件...')
    for i, parquet_file in enumerate(parquet_files):
        try:
            df = pd.read_parquet(parquet_file)
            episodes = df['episode_index'].unique()

            for ep in episodes:
                ep_data = df[df['episode_index'] == ep]
                timestamps = ep_data['timestamp'].tolist()
                timestamps = [ts[0] if isinstance(ts, (list, np.ndarray)) else ts for ts in timestamps]

                if len(timestamps) > 1:
                    time_diffs = np.diff(timestamps)
                    avg_dt = np.mean(time_diffs)
                    std_dt = np.std(time_diffs)
                    duration = timestamps[-1] - timestamps[0]
                    estimated_fps = 1.0 / avg_dt if avg_dt > 0 else 0
                else:
                    avg_dt = 0
                    std_dt = 0
                    duration = 0
                    estimated_fps = 0

                all_episodes_info[int(ep)] = {
                    'frames': len(ep_data),
                    'duration': duration,
                    'avg_dt': avg_dt,
                    'std_dt': std_dt,
                    'estimated_fps': estimated_fps,
                    'file': parquet_file.name
                }

        except Exception as e:
            print(f'❌ 读取文件 {parquet_file.name} 失败: {e}')

    # 按episode ID排序
    sorted_episodes = sorted(all_episodes_info.items())

    # 打印表格
    print(f"{'Demo ID':<8} {'帧数':<8} {'时长(s)':<10} {'FPS':<8} {'平均dt(s)':<12} {'dt标准差':<10} {'文件'}")
    print('-' * 80)

    for ep_id, info in sorted_episodes[:30]:  # 显示前30个
        print(f"{ep_id:<8} {info['frames']:<8} {info['duration']:<10.2f} {info['estimated_fps']:<8.1f} "
              f"{info['avg_dt']:<12.4f} {info['std_dt']:<10.4f} {info['file']}")

    # 统计信息
    print(f'\n📊 统计信息:')
    print(f'总Demo数量: {len(all_episodes_info)}')

    if all_episodes_info:
        frames_list = [info['frames'] for info in all_episodes_info.values()]
        durations_list = [info['duration'] for info in all_episodes_info.values()]
        dts_list = [info['avg_dt'] for info in all_episodes_info.values() if info['avg_dt'] > 0]

        print(f'帧数统计:')
        print(f'  范围: {min(frames_list)} - {max(frames_list)} 帧')
        print(f'  平均: {np.mean(frames_list):.1f} 帧')
        print(f'  中位数: {np.median(frames_list):.1f} 帧')
        print(f'  标准差: {np.std(frames_list):.1f} 帧')

        print(f'时长统计:')
        print(f'  范围: {min(durations_list):.2f} - {max(durations_list):.2f} 秒')
        print(f'  平均: {np.mean(durations_list):.2f} 秒')
        print(f'  总时长: {sum(durations_list):.2f} 秒 ({sum(durations_list)/60:.1f} 分钟)')

        if dts_list:
            print(f'时间间隔(dt)统计:')
            print(f'  平均: {np.mean(dts_list):.4f} 秒')
            print(f'  理论FPS: {1.0/np.mean(dts_list):.1f}')

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("使用方法: python detailed_analysis.py <dataset_path>")
        sys.exit(1)

    analyze_demos(sys.argv[1])