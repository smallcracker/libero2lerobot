#!/usr/bin/env python3
"""
快速分析Libero数据集的parquet文件
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import io
import json

def analyze_libero_dataset(dataset_path):
    dataset_path = Path(dataset_path).expanduser()

    # 读取info.json
    with open(dataset_path / "meta" / "info.json", 'r') as f:
        info = json.load(f)

    print("=" * 60)
    print("Libero数据集快速分析")
    print("=" * 60)

    # 基本信息
    print(f"\n📊 基本信息:")
    print(f"  总Episodes: {info['total_episodes']}")
    print(f"  总帧数: {info['total_frames']}")
    print(f"  FPS: {info['fps']}")
    print(f"  机器人类型: {info['robot_type']}")

    # 计算平均episode长度
    avg_length = info['total_frames'] / info['total_episodes']
    print(f"  平均Episode长度: {avg_length:.1f} 帧")
    print(f"  估计总时长: {info['total_frames'] / info['fps']:.1f} 秒")

    # 图像信息
    print(f"\n📷 图像信息:")
    front_shape = info['features']['observation.images.front']['shape']
    wrist_shape = info['features']['observation.images.wrist']['shape']
    print(f"  Front相机: {front_shape[0]}x{front_shape[1]} ({front_shape[2]} 通道)")
    print(f"  Wrist相机: {wrist_shape[0]}x{wrist_shape[1]} ({wrist_shape[2]} 通道)")

    # Action和State维度
    action_dim = info['features']['action']['shape'][0]
    state_dim = info['features']['observation.state']['shape'][0]
    print(f"  Action维度: {action_dim}")
    print(f"  State维度: {state_dim}")

    # 分析parquet文件
    data_dir = dataset_path / "data"
    parquet_files = list(data_dir.rglob("*.parquet"))

    print(f"\n📁 Parquet文件:")
    print(f"  文件数量: {len(parquet_files)}")

    # 读取第一个文件来分析结构
    first_file = parquet_files[0]
    df = pd.read_parquet(first_file)

    print(f"\n🔍 样本数据结构 (来自 {first_file.name}):")
    print(f"  行数: {len(df)}")
    print(f"  列: {list(df.columns)}")

    # 分析episode分布
    episodes = df['episode_index'].unique()
    print(f"  Episodes范围: {min(episodes)} - {max(episodes)}")
    print(f"  Episode数量: {len(episodes)}")

    # 分析第一个episode的详细信息
    first_episode = min(episodes)
    first_ep_data = df[df['episode_index'] == first_episode]

    if len(first_ep_data) > 1:
        timestamps = first_ep_data['timestamp'].tolist()
        timestamps = [ts[0] if isinstance(ts, (list, np.ndarray)) else ts for ts in timestamps]

        time_diffs = np.diff(timestamps)
        avg_dt = np.mean(time_diffs)
        std_dt = np.std(time_diffs)

        print(f"\n🎯 Episode {first_episode} 详细信息:")
        print(f"  帧数: {len(first_ep_data)}")
        print(f"  时间间隔: {avg_dt:.4f}s ± {std_dt:.4f}s")
        print(f"  估计FPS: {1/avg_dt:.1f}" if avg_dt > 0 else "  FPS: 无法计算")
        print(f"  持续时间: {timestamps[-1] - timestamps[0]:.2f}s")

        # 分析图像数据
        try:
            front_img_data = first_ep_data['observation.images.front'].iloc[0]
            if isinstance(front_img_data, dict) and 'bytes' in front_img_data:
                img = Image.open(io.BytesIO(front_img_data['bytes']))
                print(f"  Front图像实际尺寸: {img.size} (模式: {img.mode})")
        except Exception as e:
            print(f"  图像分析失败: {e}")

    # 分析文件大小
    total_size = sum(f.stat().st_size for f in parquet_files) / (1024 * 1024 * 1024)
    print(f"\n💾 存储信息:")
    print(f"  Parquet文件总大小: {total_size:.2f} GB")
    print(f"  平均文件大小: {total_size/len(parquet_files):.2f} GB")

    # 分析所有文件的episode分布
    all_episodes = []
    episode_lengths = {}

    for parquet_file in parquet_files[:5]:  # 只分析前5个文件以节省时间
        try:
            df = pd.read_parquet(parquet_file)
            file_episodes = df['episode_index'].unique()

            for ep in file_episodes:
                ep_length = len(df[df['episode_index'] == ep])
                episode_lengths[int(ep)] = ep_length
                all_episodes.append(int(ep))
        except Exception as e:
            print(f"  ❌ 读取 {parquet_file.name} 失败: {e}")

    if episode_lengths:
        lengths = list(episode_lengths.values())
        print(f"\n📈 Episode长度统计:")
        print(f"  长度范围: {min(lengths)} - {max(lengths)} 帧")
        print(f"  平均长度: {np.mean(lengths):.1f} 帧")
        print(f"  中位数长度: {np.median(lengths):.1f} 帧")
        print(f"  标准差: {np.std(lengths):.1f} 帧")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("使用方法: python quick_explore.py <dataset_path>")
        sys.exit(1)

    analyze_libero_dataset(sys.argv[1])