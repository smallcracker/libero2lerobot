# -*- coding: utf-8 -*-
"""
Libero统一数据转换器

支持自动识别RLDS和HDF5格式，自动解析成LeRobotDataSet v2.1格式
"""

import argparse
import json
import logging
import math
import shutil
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import jsonlines

# 检查依赖
try:
    import tensorflow_datasets as tfds
    import tensorflow as tf

    HAS_TF = True
except ImportError:
    HAS_TF = False
    logging.warning("tensorflow_datasets未安装，RLDS支持将被禁用")


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(processName)s:%(process)d] [%(filename)s:%(lineno)d] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# v2.1 版本常量
V21 = "v2.1"
DEFAULT_CHUNK_SIZE = 1000
LEGACY_DATA_PATH_TEMPLATE = (
    "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
)
LEGACY_VIDEO_PATH_TEMPLATE = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)
LEGACY_EPISODES_PATH = "meta/episodes.jsonl"
LEGACY_EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
LEGACY_TASKS_PATH = "meta/tasks.jsonl"


def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """
    将四元数转换为欧拉角 (roll, pitch, yaw)

    Args:
        quat: 四元数数组，形状为 (..., 4)，格式为 [x, y, z, w] 或 [w, x, y, z]
              根据LIBERO的惯例，假设格式为 [x, y, z, w]

    Returns:
        欧拉角数组，形状为 (..., 3)，格式为 [roll, pitch, yaw]，单位为弧度
    """
    # 假设四元数格式为 [x, y, z, w]
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    # 处理万向锁情况
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack([roll, pitch, yaw], axis=-1)


# 任务名称常量
TASK_NAME = (
    "Stack the red block on the blue block, then stack the green block on the red block"
)

logger.info("Libero RLDS/HDF5转换器初始化 - v2.1格式")


class DatasetFormatDetector:
    """数据集格式检测器"""

    @staticmethod
    def detect_format(data_path: Union[str, Path]) -> str:
        """
        自动检测数据集格式

        Args:
            data_path: 数据集路径

        Returns:
            str: 'rlds' 或 'hdf5'
        """
        data_path = Path(data_path)

        # 检查HDF5格式：查找.hdf5或.h5文件
        hdf5_files = list(data_path.rglob("*.hdf5")) + list(data_path.rglob("*.h5"))

        # 检查RLDS格式：查找tfrecord文件或dataset_info.json
        rlds_indicators = (
            list(data_path.rglob("*.tfrecord*"))
            + list(data_path.rglob("dataset_info.json"))
            + list(data_path.rglob("features.json"))
        )

        if hdf5_files and not rlds_indicators:
            logger.info(f"检测到HDF5格式，找到{len(hdf5_files)}个HDF5文件")
            return "hdf5"
        elif rlds_indicators and not hdf5_files:
            logger.info(
                f"检测到RLDS格式，找到相关文件：{[f.name for f in rlds_indicators[:3]]}"
            )
            return "rlds"
        elif hdf5_files and rlds_indicators:
            logger.warning("同时发现HDF5和RLDS文件，优先使用HDF5格式")
            return "hdf5"
        else:
            raise ValueError(f"无法检测数据格式：{data_path}")


class LeRobotDatasetV21Writer:
    """LeRobot v2.1 格式数据集写入器"""

    def __init__(
        self,
        repo_id: str,
        root: Path,
        robot_type: str = "panda",
        fps: int = 20,
        features: Dict[str, Dict[str, Any]] = None,
        use_videos: bool = True,
        chunks_size: int = DEFAULT_CHUNK_SIZE,
    ):
        self.repo_id = repo_id
        self.root = Path(root)
        self.robot_type = robot_type
        self.fps = fps
        self.features = features or {}
        self.use_videos = use_videos
        self.chunks_size = chunks_size

        # 状态跟踪
        self.episode_index = 0
        self.current_episode_frames: List[Dict[str, Any]] = []
        self.episodes_metadata: List[Dict[str, Any]] = []
        self.tasks: Dict[str, int] = {}  # task_name -> task_index
        self.total_frames = 0

        # 视频写入器
        self.video_writers: Dict[str, cv2.VideoWriter] = {}
        self.video_keys = [
            k for k, v in self.features.items() if v.get("dtype") == "video"
        ]

        # 统计信息收集器 - 用于计算全局统计
        self.stats_collector: Dict[str, List[np.ndarray]] = {}
        for key in self.features:
            if self.features[key].get("dtype") != "video":
                self.stats_collector[key] = []

        # 创建目录结构
        self._init_directories()

    def _init_directories(self):
        """初始化目录结构"""
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "meta").mkdir(exist_ok=True)
        (self.root / "data").mkdir(exist_ok=True)
        if self.use_videos:
            (self.root / "videos").mkdir(exist_ok=True)

    def _get_or_create_task_index(self, task_name: str) -> int:
        """获取或创建任务索引"""
        if task_name not in self.tasks:
            self.tasks[task_name] = len(self.tasks)
        return self.tasks[task_name]

    def add_frame(self, frame_data: Dict[str, Any]):
        """添加一帧数据"""
        self.current_episode_frames.append(frame_data.copy())

    def save_episode(self):
        """保存当前episode"""
        if not self.current_episode_frames:
            return

        episode_length = len(self.current_episode_frames)
        episode_chunk = self.episode_index // self.chunks_size

        # 获取任务信息
        task_name = self.current_episode_frames[0].get("task", TASK_NAME)
        task_index = self._get_or_create_task_index(task_name)

        # 收集统计数据
        self._collect_stats()

        # 准备parquet数据
        parquet_data = self._prepare_parquet_data(task_index)

        # 保存parquet文件
        self._save_parquet(episode_chunk, parquet_data)

        # 保存视频文件（如果使用视频）
        if self.use_videos:
            self._save_videos(episode_chunk)

        # 记录episode元数据
        self.episodes_metadata.append(
            {
                "episode_index": self.episode_index,
                "length": episode_length,
                "tasks": [task_name],
            }
        )

        self.total_frames += episode_length
        self.episode_index += 1
        self.current_episode_frames = []

        logger.debug(f"保存episode {self.episode_index - 1}, 长度: {episode_length}")

    def _collect_stats(self):
        """收集当前episode的统计数据"""
        for key in self.stats_collector:
            values = []
            for frame in self.current_episode_frames:
                if key in frame:
                    value = frame[key]
                    if isinstance(value, np.ndarray):
                        values.append(value.astype(np.float64))
            if values:
                # 将整个episode的数据堆叠起来
                stacked = np.stack(values, axis=0)
                self.stats_collector[key].append(stacked)

    def _prepare_parquet_data(self, task_index: int) -> Dict[str, List]:
        """准备parquet格式的数据"""
        data = {
            "frame_index": [],
            "timestamp": [],
            "episode_index": [],
            "index": [],
            "task_index": [],
        }

        # 初始化特征列
        for key in self.features:
            if self.features[key].get("dtype") == "video":
                # 视频特征不存储在parquet中
                continue
            data[key] = []

        for frame_idx, frame in enumerate(self.current_episode_frames):
            data["frame_index"].append(frame_idx)
            data["timestamp"].append(frame_idx / self.fps)
            data["episode_index"].append(self.episode_index)
            data["index"].append(self.total_frames + frame_idx)
            data["task_index"].append(task_index)

            for key in self.features:
                if self.features[key].get("dtype") == "video":
                    continue
                if key in frame:
                    value = frame[key]
                    if isinstance(value, np.ndarray):
                        data[key].append(value.tolist())
                    else:
                        data[key].append(value)

        return data

    def _save_parquet(self, episode_chunk: int, data: Dict[str, List]):
        """保存parquet文件"""
        parquet_path = self.root / LEGACY_DATA_PATH_TEMPLATE.format(
            episode_chunk=episode_chunk,
            episode_index=self.episode_index,
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(data)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, parquet_path)

    def _save_videos(self, episode_chunk: int):
        """保存视频文件"""
        for video_key in self.video_keys:
            video_path = self.root / LEGACY_VIDEO_PATH_TEMPLATE.format(
                episode_chunk=episode_chunk,
                video_key=video_key,
                episode_index=self.episode_index,
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)

            # 获取图像尺寸
            shape = self.features[video_key].get("shape", (256, 256, 3))
            height, width = shape[0], shape[1]

            # 使用ffmpeg友好的编码器
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (width, height))

            for frame in self.current_episode_frames:
                if video_key in frame:
                    img = frame[video_key]
                    if img.shape[2] == 3:
                        # RGB -> BGR for OpenCV
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    writer.write(img)

            writer.release()

    def finalize(self):
        """完成数据集写入，保存所有元数据"""
        logger.info("正在保存v2.1格式的元数据...")

        # 保存 info.json
        self._save_info()

        # 保存 tasks.jsonl
        self._save_tasks()

        # 保存 episodes.jsonl 和 episodes_stats.jsonl
        self._save_episodes_metadata()

        # 保存 stats.json
        self._save_stats()

        logger.info(
            f"数据集保存完成: {self.episode_index} episodes, {self.total_frames} frames"
        )

    def _save_info(self):
        """保存 info.json"""
        total_episodes = self.episode_index

        info = {
            "codebase_version": V21,
            "robot_type": self.robot_type,
            "fps": self.fps,
            "features": self.features,
            "data_path": LEGACY_DATA_PATH_TEMPLATE,
            "video_path": LEGACY_VIDEO_PATH_TEMPLATE
            if self.use_videos and self.video_keys
            else None,
            "total_episodes": total_episodes,
            "total_frames": self.total_frames,
            "total_chunks": math.ceil(total_episodes / self.chunks_size)
            if total_episodes > 0
            else 0,
            "total_videos": total_episodes * len(self.video_keys)
            if self.video_keys
            else 0,
            "chunks_size": self.chunks_size,
            "total_tasks": len(self.tasks),
        }

        info_path = self.root / "meta" / "info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)

    def _save_tasks(self):
        """保存 tasks.jsonl"""
        tasks_path = self.root / LEGACY_TASKS_PATH
        tasks_path.parent.mkdir(parents=True, exist_ok=True)

        # 按 task_index 排序
        sorted_tasks = sorted(self.tasks.items(), key=lambda x: x[1])

        with jsonlines.open(tasks_path, mode="w") as writer:
            for task_name, task_index in sorted_tasks:
                writer.write(
                    {
                        "task_index": task_index,
                        "task": task_name,
                    }
                )

    def _save_episodes_metadata(self):
        """保存 episodes.jsonl 和 episodes_stats.jsonl"""
        episodes_path = self.root / LEGACY_EPISODES_PATH
        stats_path = self.root / LEGACY_EPISODES_STATS_PATH
        episodes_path.parent.mkdir(parents=True, exist_ok=True)

        with jsonlines.open(episodes_path, mode="w") as episodes_writer:
            with jsonlines.open(stats_path, mode="w") as stats_writer:
                for metadata in self.episodes_metadata:
                    episodes_writer.write(metadata)
                    # 简单的统计信息
                    stats_writer.write(
                        {
                            "episode_index": metadata["episode_index"],
                            "stats": {},
                        }
                    )

    def _save_stats(self):
        """保存 stats.json - 全局统计信息"""
        stats = {}

        for key, data_list in self.stats_collector.items():
            if not data_list:
                continue

            # 将所有episode的数据合并
            all_data = np.concatenate(data_list, axis=0)

            # 计算统计信息
            feature_stats = {
                "min": all_data.min(axis=0).tolist(),
                "max": all_data.max(axis=0).tolist(),
                "mean": all_data.mean(axis=0).tolist(),
                "std": all_data.std(axis=0).tolist(),
            }

            # 如果是标量，展开为单个值
            if (
                isinstance(feature_stats["min"], list)
                and len(feature_stats["min"]) == 1
            ):
                for stat_key in feature_stats:
                    if isinstance(feature_stats[stat_key], list):
                        feature_stats[stat_key] = feature_stats[stat_key]

            stats[key] = feature_stats

        stats_path = self.root / "meta" / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)

        logger.info(f"已保存统计信息: {list(stats.keys())}")


class HDF5Processor:
    """HDF5数据处理器"""

    def __init__(
        self, image_size: Tuple[int, int] = (256, 256), use_videos: bool = False
    ):
        self.image_size = image_size  # (height, width)
        self.use_videos = use_videos

    def get_default_features(
        self, use_videos: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """获取Libero数据集的默认特征配置"""
        image_dtype = "video" if use_videos else "image"

        return {
            "observation.images.front": {
                "dtype": image_dtype,
                "shape": (*self.image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist": {
                "dtype": image_dtype,
                "shape": (*self.image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),
                "names": [f"state_{i}" for i in range(8)],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": [f"action_{i}" for i in range(7)],
            },
        }

    def process_episode(
        self, episode_path: Path, dataset: LeRobotDatasetV21Writer, task_name: str
    ) -> bool:
        """处理单个episode数据"""
        try:
            with h5py.File(episode_path, "r") as file:
                logger.info(f"HDF5文件键: {list(file.keys())}")

                if "data" in file:
                    return self._process_libero_demo_format(file, dataset, episode_path)
                else:
                    logger.warning(f"未识别的HDF5格式: {episode_path}")
                    return False

        except (FileNotFoundError, OSError, KeyError) as e:
            logger.error(f"跳过 {episode_path}: {str(e)}")
            return False

    def _process_libero_demo_format(
        self,
        file: h5py.File,
        dataset: LeRobotDatasetV21Writer,
        file_path: Optional[Path] = None,
    ) -> bool:
        """处理新的Libero demo格式：data/demo_N/..."""
        data_group = file["data"]

        demo_keys = [k for k in data_group.keys() if k.startswith("demo_")]
        demo_keys.sort(key=lambda x: int(x.split("_")[1]))

        for demo_key in demo_keys:
            demo_group = data_group[demo_key]
            logger.info(f"处理 {demo_key}")

            demo_task_str = TASK_NAME

            actions = np.array(demo_group["actions"])
            obs_group = demo_group["obs"]

            if "eef_pos" not in obs_group:
                raise KeyError("未找到末端执行器位置数据 (eef_pos)")
            if "eef_quat" not in obs_group:
                raise KeyError("未找到末端执行器姿态数据 (eef_quat)")
            if "gripper_pos" not in obs_group:
                raise KeyError("未找到夹爪状态数据 (gripper_pos)")

            eef_pos = np.array(obs_group["eef_pos"])
            eef_quat = np.array(obs_group["eef_quat"])
            gripper_pos = np.array(obs_group["gripper_pos"])

            eef_euler = quat_to_euler(eef_quat)
            joint_states = np.concatenate([eef_pos, eef_euler, gripper_pos], axis=-1)
            logger.info(f"状态向量形状: {joint_states.shape}")

            if "agentview_rgb" in obs_group and "eye_in_hand_rgb" in obs_group:
                agentview_rgb = np.array(obs_group["agentview_rgb"])
                eye_in_hand_rgb = np.array(obs_group["eye_in_hand_rgb"])
            elif "table_cam" in obs_group and "wrist_cam" in obs_group:
                agentview_rgb = np.array(obs_group["table_cam"])
                eye_in_hand_rgb = np.array(obs_group["wrist_cam"])
            else:
                raise KeyError("未找到图像数据")

            num_frames = min(
                len(actions),
                len(joint_states),
                len(agentview_rgb),
                len(eye_in_hand_rgb),
            )

            logger.info(
                f"Demo {demo_key}: actions={len(actions)}, states={len(joint_states)}, front={len(agentview_rgb)}, wrist={len(eye_in_hand_rgb)}"
            )

            for i in tqdm(range(num_frames), desc=f"处理 {demo_key}", leave=False):
                front_img = cv2.resize(
                    agentview_rgb[i], (self.image_size[1], self.image_size[0])
                )
                wrist_img = cv2.resize(
                    eye_in_hand_rgb[i], (self.image_size[1], self.image_size[0])
                )

                frame_data = {
                    "action": actions[i].astype(np.float32),
                    "observation.state": joint_states[i].astype(np.float32),
                    "observation.images.front": front_img,
                    "observation.images.wrist": wrist_img,
                    "task": demo_task_str,
                }

                dataset.add_frame(frame_data)

            dataset.save_episode()

        return True


class RLDSProcessor:
    """RLDS数据处理器"""

    def __init__(self):
        if not HAS_TF:
            raise ImportError("tensorflow_datasets是RLDS处理所必需的")
        self.image_size = (256, 256)
        self.use_videos = False

    def get_default_features(
        self, use_videos: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """获取Libero数据集的默认特征配置"""
        image_dtype = "video" if use_videos else "image"

        return {
            "observation.images.front": {
                "dtype": image_dtype,
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist": {
                "dtype": image_dtype,
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),
                "names": [f"state_{i}" for i in range(8)],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": [f"action_{i}" for i in range(7)],
            },
        }

    def process_dataset(
        self, dataset: LeRobotDatasetV21Writer, data_source: Union[str, Path]
    ):
        """处理RLDS数据集"""
        raw_dataset_names = [
            "libero_10_no_noops",
            "libero_goal_no_noops",
            "libero_object_no_noops",
            "libero_spatial_no_noops",
        ]

        episode_idx = 0

        for raw_dataset_name in raw_dataset_names:
            logger.info(f"处理RLDS数据集: {raw_dataset_name}")

            try:
                raw_dataset = tfds.load(
                    raw_dataset_name, data_dir=data_source, split="train", try_gcs=False
                )

                for episode in raw_dataset:
                    logger.info(f"处理episode {episode_idx + 1}")

                    steps_list = list(episode["steps"].as_numpy_iterator())
                    task_str = f"episode_{episode_idx}"

                    if steps_list and "language_instruction" in steps_list[0]:
                        task_str = steps_list[0]["language_instruction"].decode()

                    for step_idx, step in enumerate(steps_list):
                        frame_data = {
                            "observation.images.front": step["observation"]["image"],
                            "observation.images.wrist": step["observation"][
                                "wrist_image"
                            ],
                            "observation.state": step["observation"]["state"].astype(
                                np.float32
                            ),
                            "action": step["action"].astype(np.float32),
                            "task": task_str,
                        }
                        dataset.add_frame(frame_data)

                    dataset.save_episode()
                    episode_idx += 1

            except Exception as e:
                logger.warning(f"处理数据集 {raw_dataset_name} 时出错: {e}")
                continue


class UnifiedConverter:
    """统一转换器类 - 生成v2.1格式"""

    def __init__(self):
        self.detector = DatasetFormatDetector()
        logger.info("转换器初始化完成 - 生成v2.1格式数据集")

    def convert_dataset(
        self,
        data_dir: Union[str, Path],
        repo_id: str,
        output_dir: Optional[Union[str, Path]] = None,
        push_to_hub: bool = False,
        use_videos: bool = True,
        robot_type: str = "panda",
        fps: int = 20,
        task_name: str = "default_task",
        hub_config: Optional[Dict[str, Any]] = None,
        clean_existing: bool = False,
        **kwargs,
    ) -> LeRobotDatasetV21Writer:
        """
        统一转换接口 - 生成v2.1格式

        Returns:
            LeRobotDatasetV21Writer: 数据集写入器
        """
        data_path = Path(data_dir)

        # 自动检测格式
        format_type = self.detector.detect_format(data_path)
        logger.info(f"检测到数据格式: {format_type}")

        # 根据格式选择处理器和特征
        if format_type == "hdf5":
            processor = HDF5Processor()
            features = processor.get_default_features(use_videos)
        else:
            processor = RLDSProcessor()
            features = processor.get_default_features(use_videos)

        # 设置输出路径
        if output_dir is None:
            lerobot_root = Path("~/.cache/huggingface/lerobot/").expanduser()
        else:
            lerobot_root = Path(output_dir).expanduser()

        os.environ["HF_LEROBOT_HOME"] = str(lerobot_root)
        lerobot_dataset_dir = lerobot_root / repo_id

        # 检查现有数据集
        if clean_existing and lerobot_dataset_dir.exists():
            logger.info(f"清理现有数据集: {lerobot_dataset_dir}")
            shutil.rmtree(lerobot_dataset_dir)

        lerobot_root.mkdir(parents=True, exist_ok=True)

        # 创建v2.1格式的数据集写入器
        logger.info(f"创建LeRobot v2.1数据集: {repo_id}")
        logger.info(f"机器人类型: {robot_type}, 帧率: {fps}, 使用视频: {use_videos}")

        dataset = LeRobotDatasetV21Writer(
            repo_id=repo_id,
            root=lerobot_dataset_dir,
            robot_type=robot_type,
            fps=fps,
            features=features,
            use_videos=use_videos,
        )

        # 处理数据
        if format_type == "hdf5":
            self._process_hdf5_data(processor, dataset, data_path, TASK_NAME)
        else:
            if isinstance(processor, RLDSProcessor):
                processor.process_dataset(dataset, data_path)
            else:
                raise TypeError("RLDS格式需要RLDSProcessor实例")

        # 完成写入，保存所有元数据
        dataset.finalize()

        logger.info("✅ v2.1格式数据集转换完成!")
        return dataset

    def _process_hdf5_data(
        self,
        processor: HDF5Processor,
        dataset: LeRobotDatasetV21Writer,
        data_path: Path,
        task_name: str,
    ):
        """单线程处理HDF5数据"""
        # 查找所有episode
        episodes = []
        for ep_dir in data_path.iterdir():
            if ep_dir.is_dir():
                ep_path = ep_dir / "data" / "trajectory.hdf5"
                if ep_path.exists():
                    episodes.append(ep_path)

        if not episodes:
            episodes = list(data_path.rglob("*.hdf5")) + list(data_path.rglob("*.h5"))

        logger.info(f"找到 {len(episodes)} 个episode文件")

        for ep_path in tqdm(episodes, desc="处理Episodes"):
            processor.process_episode(ep_path, dataset, TASK_NAME)
            logger.info(f"处理完成: {ep_path.name}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Libero统一数据转换器 - 生成LeRobot v2.1格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--data-dir", type=str, required=True, help="数据目录路径")
    parser.add_argument("--repo-id", type=str, required=True, help="数据集仓库ID")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--push-to-hub", action="store_true", help="推送到Hub")
    parser.add_argument("--private", action="store_true", help="创建私有数据集")
    parser.add_argument("--use-videos", action="store_true", help="使用视频格式")
    parser.add_argument("--robot-type", type=str, default="panda", help="机器人类型")
    parser.add_argument("--fps", type=int, default=20, help="帧率")
    parser.add_argument(
        "--task-name", type=str, default="default_task", help="任务名称"
    )
    parser.add_argument("--clean-existing", action="store_true", help="清理现有数据集")
    parser.add_argument("--verbose", action="store_true", help="详细日志")
    parser.add_argument("--dry-run", action="store_true", help="试运行模式")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not Path(args.data_dir).exists():
        logger.error(f"数据目录不存在: {args.data_dir}")
        return 1

    if "/" not in args.repo_id:
        logger.error(f"repo_id格式错误: {args.repo_id}")
        return 1

    logger.info("📋 转换配置 (v2.1格式):")
    logger.info(f"  数据源: {args.data_dir}")
    logger.info(f"  仓库ID: {args.repo_id}")
    logger.info(f"  使用视频: {args.use_videos}")

    if args.dry_run:
        logger.info("✅ 试运行完成，参数验证通过")
        return 0

    try:
        converter = UnifiedConverter()

        dataset = converter.convert_dataset(
            data_dir=args.data_dir,
            repo_id=args.repo_id,
            output_dir=args.output_dir,
            push_to_hub=args.push_to_hub,
            use_videos=args.use_videos,
            robot_type=args.robot_type,
            fps=args.fps,
            task_name=TASK_NAME,
            clean_existing=args.clean_existing,
        )

        logger.info("✅ 转换完成!")
        return 0

    except Exception as e:
        logger.error(f"转换失败: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
