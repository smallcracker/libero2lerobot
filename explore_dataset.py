#!/usr/bin/env python3
"""
Libero数据集探索工具
分析LeRobot格式的Libero数据集，提供详细的统计信息和可视化
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import io
import argparse
from typing import Dict, List, Tuple, Any
from collections import defaultdict

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("错误: 请安装LeRobot: pip install lerobot")
    sys.exit(1)


def analyze_dataset_basic_info(dataset_path: str) -> Dict[str, Any]:
    """分析数据集基本信息"""
    dataset_path = Path(dataset_path).expanduser()

    if not dataset_path.exists():
        return {"error": f"数据集路径不存在: {dataset_path}"}

    # 读取info.json
    info_file = dataset_path / "meta" / "info.json"
    if not info_file.exists():
        return {"error": "找不到meta/info.json文件"}

    with open(info_file, "r", encoding="utf-8") as f:
        info = json.load(f)

    # 尝试加载LeRobot数据集
    try:
        dataset = LeRobotDataset(repo_id="", root=str(dataset_path))
        total_frames = len(dataset)

        # 获取episode统计
        episode_indices = dataset.episode_data_index["episode_index"].tolist()
        total_episodes = len(set(episode_indices))

        # 计算平均episode长度
        episode_lengths = []
        for ep_idx in set(episode_indices):
            ep_mask = episode_indices == ep_idx
            episode_lengths.append(np.sum(ep_mask))

        avg_length = np.mean(episode_lengths) if episode_lengths else 0
        min_length = np.min(episode_lengths) if episode_lengths else 0
        max_length = np.max(episode_lengths) if episode_lengths else 0

        return {
            "dataset_path": str(dataset_path),
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "avg_episode_length": avg_length,
            "min_episode_length": min_length,
            "max_episode_length": max_length,
            "fps": info.get("fps", "unknown"),
            "features": info.get("features", {}),
            "splits": info.get("splits", {}),
            "dataset_size_mb": info.get("data_files_size_in_mb", "unknown"),
        }

    except Exception as e:
        return {
            "error": f"加载数据集失败: {str(e)}",
            "dataset_path": str(dataset_path),
            "info": info,
        }


def analyze_demo_details(dataset_path: str, max_demos: int = 10) -> Dict[str, Any]:
    """详细分析demo信息"""
    dataset_path = Path(dataset_path).expanduser()

    try:
        dataset = LeRobotDataset(repo_id="", root=str(dataset_path))

        demo_info = []
        episode_indices = dataset.episode_data_index["episode_index"].tolist()
        unique_episodes = sorted(set(episode_indices))

        # 限制分析的demo数量
        episodes_to_analyze = unique_episodes[: min(max_demos, len(unique_episodes))]

        for ep_idx in episodes_to_analyze:
            # 获取该episode的所有帧索引
            ep_mask = np.array(episode_indices) == ep_idx
            frame_indices = np.where(ep_mask)[0]

            if len(frame_indices) == 0:
                continue

            # 获取第一帧和最后一帧的时间戳来计算dt
            first_frame_idx = frame_indices[0]
            last_frame_idx = frame_indices[-1]

            # 获取时间戳信息
            timestamps = dataset.frames["timestamp"][frame_indices]

            if len(timestamps) > 1:
                dt_list = np.diff(timestamps.flatten())
                avg_dt = np.mean(dt_list)
                std_dt = np.std(dt_list)
            else:
                avg_dt = 0
                std_dt = 0

            # 分析图像尺寸
            try:
                # 获取第一帧的图像
                front_img_data = dataset.frames["observation.images.front"][
                    first_frame_idx
                ]
                if isinstance(front_img_data, dict) and "bytes" in front_img_data:
                    img = Image.open(io.BytesIO(front_img_data["bytes"]))
                    image_size = img.size  # (width, height)
                    image_mode = img.mode
                else:
                    # 如果是直接存储的numpy数组
                    image_size = (
                        front_img_data.shape[:2][::-1]
                        if hasattr(front_img_data, "shape")
                        else "unknown"
                    )
                    image_mode = "array"
            except:
                image_size = "unknown"
                image_mode = "unknown"

            # 获取action和state的维度
            try:
                action_sample = dataset.frames["action"][first_frame_idx]
                action_dim = (
                    action_sample.shape[-1]
                    if hasattr(action_sample, "shape")
                    else len(action_sample)
                )
            except:
                action_dim = "unknown"

            try:
                state_sample = dataset.frames["observation.state"][first_frame_idx]
                state_dim = (
                    state_sample.shape[-1]
                    if hasattr(state_sample, "shape")
                    else len(state_sample)
                )
            except:
                state_dim = "unknown"

            demo_info.append(
                {
                    "demo_id": int(ep_idx),
                    "num_frames": len(frame_indices),
                    "avg_dt": float(avg_dt),
                    "std_dt": float(std_dt),
                    "estimated_fps": 1.0 / avg_dt if avg_dt > 0 else 0,
                    "image_size": image_size,
                    "image_mode": image_mode,
                    "action_dim": action_dim,
                    "state_dim": state_dim,
                    "duration_seconds": float(timestamps[-1][0] - timestamps[0][0])
                    if len(timestamps) > 0
                    else 0,
                }
            )

        return {
            "analyzed_demos": len(demo_info),
            "total_demos": len(unique_episodes),
            "demo_details": demo_info,
        }

    except Exception as e:
        return {"error": f"分析demo详情失败: {str(e)}"}


def analyze_parquet_files(dataset_path: str) -> Dict[str, Any]:
    """分析parquet文件结构"""
    dataset_path = Path(dataset_path).expanduser()
    data_dir = dataset_path / "data"

    if not data_dir.exists():
        return {"error": "找不到data目录"}

    parquet_files = list(data_dir.rglob("*.parquet"))

    if not parquet_files:
        return {"error": "找不到parquet文件"}

    file_info = []
    total_rows = 0

    for parquet_file in sorted(parquet_files):
        try:
            df = pd.read_parquet(parquet_file)

            # 分析每列的数据类型和大小
            column_info = {}
            for col in df.columns:
                if (
                    col == "observation.images.front"
                    or col == "observation.images.wrist"
                ):
                    # 图像列的特殊分析
                    sample = df[col].iloc[0] if len(df) > 0 else None
                    if isinstance(sample, dict) and "bytes" in sample:
                        column_info[col] = {
                            "dtype": str(df[col].dtype),
                            "sample_size_bytes": len(sample["bytes"]),
                            "format": "image_binary",
                        }
                    else:
                        column_info[col] = {
                            "dtype": str(df[col].dtype),
                            "shape": str(df[col].iloc[0].shape)
                            if hasattr(df[col].iloc[0], "shape")
                            else "unknown",
                            "format": "array",
                        }
                else:
                    column_info[col] = {
                        "dtype": str(df[col].dtype),
                        "shape": str(df[col].iloc[0].shape)
                        if len(df) > 0 and hasattr(df[col].iloc[0], "shape")
                        else "scalar",
                    }

            file_info.append(
                {
                    "file_path": str(parquet_file.relative_to(dataset_path)),
                    "rows": len(df),
                    "columns": list(df.columns),
                    "file_size_mb": parquet_file.stat().st_size / (1024 * 1024),
                    "column_details": column_info,
                }
            )

            total_rows += len(df)

        except Exception as e:
            file_info.append(
                {
                    "file_path": str(parquet_file.relative_to(dataset_path)),
                    "error": str(e),
                }
            )

    return {
        "total_parquet_files": len(parquet_files),
        "total_rows": total_rows,
        "files": file_info,
    }


def print_summary(dataset_path: str):
    """打印数据集摘要信息"""
    print("=" * 60)
    print("Libero数据集分析报告")
    print("=" * 60)

    # 基本信息
    basic_info = analyze_dataset_basic_info(dataset_path)
    if "error" in basic_info:
        print(f"❌ 错误: {basic_info['error']}")
        return

    print(f"\n📊 基本信息:")
    print(f"  数据集路径: {basic_info['dataset_path']}")
    print(f"  总Episodes: {basic_info['total_episodes']}")
    print(f"  总帧数: {basic_info['total_frames']}")
    print(f"  平均Episode长度: {basic_info['avg_episode_length']:.1f} 帧")
    print(
        f"  Episode长度范围: {basic_info['min_episode_length']} - {basic_info['max_episode_length']} 帧"
    )
    print(f"  FPS: {basic_info['fps']}")
    print(f"  数据集大小: {basic_info['dataset_size_mb']} MB")

    # Demo详情
    print(f"\n🔍 Demo详情 (前5个):")
    demo_details = analyze_demo_details(dataset_path, max_demos=5)

    if "error" not in demo_details:
        for demo in demo_details["demo_details"]:
            print(f"  Demo {demo['demo_id']}:")
            print(f"    帧数: {demo['num_frames']}")
            print(f"    平均dt: {demo['avg_dt']:.4f}s ± {demo['std_dt']:.4f}s")
            print(f"    估计FPS: {demo['estimated_fps']:.1f}")
            print(f"    持续时间: {demo['duration_seconds']:.2f}s")
            print(f"    图像尺寸: {demo['image_size']}")
            print(f"    Action维度: {demo['action_dim']}")
            print(f"    State维度: {demo['state_dim']}")
            print()

    # Parquet文件分析
    print(f"📁 Parquet文件结构:")
    parquet_info = analyze_parquet_files(dataset_path)

    if "error" not in parquet_info:
        print(f"  总文件数: {parquet_info['total_parquet_files']}")
        print(f"  总行数: {parquet_info['total_rows']}")
        print(f"  文件详情:")

        for file in parquet_info["files"]:
            if "error" in file:
                print(f"    ❌ {file['file_path']}: {file['error']}")
            else:
                print(f"    📄 {file['file_path']}:")
                print(f"      行数: {file['rows']}")
                print(f"      文件大小: {file['file_size_mb']:.2f} MB")
                print(f"      列: {', '.join(file['columns'])}")


def main():
    parser = argparse.ArgumentParser(description="探索Libero数据集")
    parser.add_argument("dataset_path", help="LeRobot数据集路径")
    parser.add_argument("--max-demos", type=int, default=10, help="分析的最大demo数量")
    parser.add_argument("--output", help="输出JSON文件路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")

    args = parser.parse_args()

    # 打印摘要
    print_summary(args.dataset_path)

    # 如果需要输出到文件
    if args.output:
        analysis_result = {
            "basic_info": analyze_dataset_basic_info(args.dataset_path),
            "demo_details": analyze_demo_details(args.dataset_path, args.max_demos),
            "parquet_analysis": analyze_parquet_files(args.dataset_path),
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)

        print(f"\n💾 详细分析结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
