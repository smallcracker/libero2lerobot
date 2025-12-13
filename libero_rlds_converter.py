# -*- coding: utf-8 -*-
"""
Libero统一数据转换器

支持自动识别RLDS和HDF5格式，自动解析成LeRobotDataSet格式，支持多线程操作
"""

import argparse
import ast
import json
import logging
import shutil
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
import concurrent.futures
from functools import partial
import multiprocessing
import sys

import cv2
import h5py
import numpy as np
from tqdm import tqdm

# 检查依赖
try:
    import tensorflow_datasets as tfds
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    logging.warning("tensorflow_datasets未安装，RLDS支持将被禁用")

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    # from lerobot.common.constants import HF_LEROBOT_HOME
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    # from lerobot.constants import HF_LEROBOT_HOME


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(processName)s:%(process)d] [%(filename)s:%(lineno)d] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# 任务名称常量
TASK_NAME = "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0"

logger.info("Libero RLDS/HDF5转换器初始化")



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
            list(data_path.rglob("*.tfrecord*")) + 
            list(data_path.rglob("dataset_info.json")) +
            list(data_path.rglob("features.json"))
        )
        
        if hdf5_files and not rlds_indicators:
            logger.info(f"检测到HDF5格式，找到{len(hdf5_files)}个HDF5文件")
            return "hdf5"
        elif rlds_indicators and not hdf5_files:
            logger.info(f"检测到RLDS格式，找到相关文件："
                       f"{[f.name for f in rlds_indicators[:3]]}")
            return "rlds"
        elif hdf5_files and rlds_indicators:
            logger.warning("同时发现HDF5和RLDS文件，优先使用HDF5格式")
            return "hdf5"
        else:
            raise ValueError(f"无法检测数据格式：{data_path}")


class HDF5Processor:
    """HDF5数据处理器"""
    
    def __init__(self, image_size: Tuple[int, int] = (256, 256), 
                 use_videos: bool = False):
        """
        初始化HDF5处理器
        
        Args:
            image_size: 图像尺寸 (height, width) - 匹配numpy数组格式
            use_videos: 是否使用视频格式
        """
        self.image_size = image_size  # (height, width)
        self.use_videos = use_videos
    
    def get_default_features(self, use_videos: bool = True) -> Dict[str, Dict[str, Any]]:
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
    
    def process_episode(self, episode_path: Path, dataset: LeRobotDataset, task_name: str) -> bool:
        """
        处理单个episode数据
        
        Args:
            episode_path: episode文件路径
            dataset: LeRobot数据集
            task_name: 任务名称（已弃用，现在使用TASK_NAME常量）
            
        Returns:
            bool: 处理是否成功
        """
        try:
            with h5py.File(episode_path, "r") as file:
                logger.debug(f"HDF5文件键: {list(file.keys())}")
                
                # 检测HDF5文件格式，支持多种Libero格式
                if "data" in file:
                    # 新的Libero格式：data/demo_N/...
                    return self._process_libero_demo_format(file, dataset, task_name, episode_path)
                else:
                    logger.warning(f"未识别的HDF5格式: {episode_path}")
                    return False

        except (FileNotFoundError, OSError, KeyError) as e:
            logger.error(f"跳过 {episode_path}: {str(e)}")
            return False
            

    def _process_libero_demo_format(self, file: h5py.File, dataset: LeRobotDataset, task_name: str, file_path: Optional[Path] = None) -> bool:
        """处理新的Libero demo格式：data/demo_N/..."""
        # 注意：task_name参数已弃用，现在使用TASK_NAME常量
        data_group = file["data"]
        
        # 使用常量任务名称
        task_str = TASK_NAME
        
        # 获取所有demo
        demo_keys = [k for k in data_group.keys() if k.startswith("demo_")]
        demo_keys.sort(key=lambda x: int(x.split("_")[1]))  # 按数字排序
        
        for demo_key in demo_keys:
            demo_group = data_group[demo_key]
            logger.info(f"处理 {demo_key}")
            
            # 使用常量任务名称
            demo_task_str = TASK_NAME
            
            # 读取动作数据
            actions = np.array(demo_group["actions"])
            
            # 读取观察数据
            obs_group = demo_group["obs"]
            
            # 读取关节状态 - 作为observation.state
            # 尝试多种可能的关节状态字段名
            if "joint_states" in obs_group:
                joint_states = np.array(obs_group["joint_states"])
            elif "joint_pos" in obs_group:
                joint_states = np.array(obs_group["joint_pos"])
            else:
                raise KeyError("未找到关节状态数据 (joint_states 或 joint_pos)")
            
            # 调整状态维度以匹配RLDS格式
            if joint_states.shape[-1] == 7:
                # 添加一个额外的维度（例如夹爪状态，设为0）
                gripper_state = np.zeros((joint_states.shape[0], 1), dtype=joint_states.dtype)
                joint_states = np.concatenate([joint_states, gripper_state], axis=-1)
            elif joint_states.shape[-1] == 9:
                # 如果是9维，取前7维作为关节位置，后2维可能是夹爪状态
                # 我们取前7维，然后添加一个额外的维度
                joint_states = joint_states[:, :7]
                gripper_state = np.zeros((joint_states.shape[0], 1), dtype=joint_states.dtype)
                joint_states = np.concatenate([joint_states, gripper_state], axis=-1)
            
            # 读取图像数据 - 尝试多种可能的字段名
            if "agentview_rgb" in obs_group and "eye_in_hand_rgb" in obs_group:
                agentview_rgb = np.array(obs_group["agentview_rgb"])  # 前视图像
                eye_in_hand_rgb = np.array(obs_group["eye_in_hand_rgb"])  # 腕部图像
            elif "table_cam" in obs_group and "wrist_cam" in obs_group:
                agentview_rgb = np.array(obs_group["table_cam"])  # 前视图像
                eye_in_hand_rgb = np.array(obs_group["wrist_cam"])  # 腕部图像
            else:
                raise KeyError("未找到图像数据 (agentview_rgb/eye_in_hand_rgb 或 table_cam/wrist_cam)")
            
            # 确保所有数组长度一致
            num_frames = min(len(actions), len(joint_states), len(agentview_rgb), len(eye_in_hand_rgb))
            
            # 处理每一帧
            for i in tqdm(range(num_frames), desc=f"处理 {demo_key}", leave=False):
                # 处理图像：调整大小到目标尺寸
                front_img = cv2.resize(agentview_rgb[i], (self.image_size[1], self.image_size[0]))
                wrist_img = cv2.resize(eye_in_hand_rgb[i], (self.image_size[1], self.image_size[0]))
                
                # 修复图像旋转问题：翻转180度
                front_img = cv2.flip(front_img, -1)  # -1表示180度翻转
                wrist_img = cv2.flip(wrist_img, -1)  # -1表示180度翻转
                
                # 准备帧数据 - 参考RLDS格式
                frame_data = {
                    "action": actions[i].astype(np.float32),
                    "observation.state": joint_states[i].astype(np.float32),
                    "observation.images.front": front_img,
                    "observation.images.wrist": wrist_img,
                    "task": demo_task_str,
                }
                
                # 添加帧到数据集 - 修复API调用
                logger.debug(f"Adding frame with keys: {frame_data.keys()}")
                dataset.add_frame(frame_data)
            
            # 每个demo保存为一个episode
            dataset.save_episode()
        
        return True


    def _extract_libero_demo_frames(self, file: h5py.File, task_name: str, file_path: Optional[Path] = None) -> List[List[Dict]]:
        """提取Libero demo格式的帧数据（用于多线程处理）- 返回按demo分组的数据"""
        # 注意：task_name参数已弃用，现在使用TASK_NAME常量
        demos_frames = []
        data_group = file["data"]
        
        # 使用常量任务名称
        task_str = TASK_NAME
        
        # 获取所有demo
        demo_keys = [k for k in data_group.keys() if k.startswith("demo_")]
        demo_keys.sort(key=lambda x: int(x.split("_")[1]))  # 按数字排序
        
        for demo_key in demo_keys:
            demo_frames = []
            demo_group = data_group[demo_key]
            
            # 使用常量任务名称
            demo_task_str = TASK_NAME
            
            # 读取动作数据
            actions = np.array(demo_group["actions"])
            
            # 读取观察数据
            obs_group = demo_group["obs"]
            
            # 读取关节状态 - 作为observation.state
            # 尝试多种可能的关节状态字段名
            if "joint_states" in obs_group:
                joint_states = np.array(obs_group["joint_states"])
            elif "joint_pos" in obs_group:
                joint_states = np.array(obs_group["joint_pos"])
            else:
                raise KeyError("未找到关节状态数据 (joint_states 或 joint_pos)")
            
            # 调整状态维度以匹配RLDS格式
            if joint_states.shape[-1] == 7:
                # 添加一个额外的维度（例如夹爪状态，设为0）
                gripper_state = np.zeros((joint_states.shape[0], 1), dtype=joint_states.dtype)
                joint_states = np.concatenate([joint_states, gripper_state], axis=-1)
            elif joint_states.shape[-1] == 9:
                # 如果是9维，取前7维作为关节位置，后2维可能是夹爪状态
                # 我们取前7维，然后添加一个额外的维度
                joint_states = joint_states[:, :7]
                gripper_state = np.zeros((joint_states.shape[0], 1), dtype=joint_states.dtype)
                joint_states = np.concatenate([joint_states, gripper_state], axis=-1)
            
            # 读取图像数据 - 尝试多种可能的字段名
            if "agentview_rgb" in obs_group and "eye_in_hand_rgb" in obs_group:
                agentview_rgb = np.array(obs_group["agentview_rgb"])  # 前视图像
                eye_in_hand_rgb = np.array(obs_group["eye_in_hand_rgb"])  # 腕部图像
            elif "table_cam" in obs_group and "wrist_cam" in obs_group:
                agentview_rgb = np.array(obs_group["table_cam"])  # 前视图像
                eye_in_hand_rgb = np.array(obs_group["wrist_cam"])  # 腕部图像
            else:
                raise KeyError("未找到图像数据 (agentview_rgb/eye_in_hand_rgb 或 table_cam/wrist_cam)")
            
            # 确保所有数组长度一致
            num_frames = min(len(actions), len(joint_states), len(agentview_rgb), len(eye_in_hand_rgb))
            
            # 处理每一帧
            for i in range(num_frames):
                # 处理图像：调整大小到目标尺寸
                front_img = cv2.resize(agentview_rgb[i], (self.image_size[1], self.image_size[0]))
                wrist_img = cv2.resize(eye_in_hand_rgb[i], (self.image_size[1], self.image_size[0]))
                
                # 修复图像旋转问题：翻转180度
                front_img = cv2.flip(front_img, -1)  # -1表示180度翻转
                wrist_img = cv2.flip(wrist_img, -1)  # -1表示180度翻转
                
                # 准备帧数据 - 参考RLDS格式
                frame_data = {
                    "action": actions[i].astype(np.float32),
                    "observation.state": joint_states[i].astype(np.float32),
                    "observation.images.front": front_img,
                    "observation.images.wrist": wrist_img,
                    "task": demo_task_str,
                }
                demo_frames.append(frame_data)
            
            demos_frames.append(demo_frames)
        
        return demos_frames

    def _extract_direct_episode_frames(self, file: h5py.File, task_name: str, file_path: Optional[Path] = None) -> List[List[Dict]]:
        """提取直接episode格式的帧数据（兜底方案，用于多线程处理）- 返回按episode分组的数据"""
        # 注意：task_name参数已弃用，现在使用TASK_NAME常量
        # 兜底方案：将整个文件作为一个episode
        episode_frames = []
        
        # 使用常量任务名称
        task_str = TASK_NAME
        
        # 尝试找到可能的状态数据
        state_keys = ["joint_states", "joint_pos", "states", "robot_states", "state"]
        state_data = None
        
        for key in state_keys:
            if key in file:
                state_data = np.array(file[key])
                logger.info(f"找到状态数据: {key}, shape: {state_data.shape}")
                break
        
        if state_data is None:
            raise KeyError("未找到任何状态数据")
        
        # 调整状态维度以匹配RLDS格式
        if state_data.shape[-1] == 7:
            # 添加一个额外的维度（例如夹爪状态，设为0）
            gripper_state = np.zeros((state_data.shape[0], 1), dtype=state_data.dtype)
            state_data = np.concatenate([state_data, gripper_state], axis=-1)
            logger.info(f"状态数据从7维扩展为8维: {state_data.shape}")
        elif state_data.shape[-1] == 9:
            # 如果是9维，取前7维作为关节位置，后2维可能是夹爪状态
            # 我们取前7维，然后添加一个额外的维度
            state_data = state_data[:, :7]
            gripper_state = np.zeros((state_data.shape[0], 1), dtype=state_data.dtype)
            state_data = np.concatenate([state_data, gripper_state], axis=-1)
            logger.info(f"状态数据从9维调整为8维: {state_data.shape}")
        
        # 尝试找到动作数据
        action_keys = ["actions", "action"]
        action_data = None
        
        for key in action_keys:
            if key in file:
                action_data = np.array(file[key])
                break
        
        if action_data is None:
            logger.warning("未找到动作数据，使用状态数据作为动作")
            action_data = state_data
        
        # 尝试找到图像数据
        image_keys = ["agentview_rgb", "table_cam", "images", "rgb"]
        front_image_data = None
        wrist_image_data = None
        
        # 先尝试找到前视图像
        for key in image_keys:
            if key in file:
                front_image_data = np.array(file[key])
                logger.info(f"找到前视图像数据: {key}, shape: {front_image_data.shape}")
                break
        
        # 尝试找到腕部图像
        wrist_image_keys = ["eye_in_hand_rgb", "wrist_cam"]
        for key in wrist_image_keys:
            if key in file:
                wrist_image_data = np.array(file[key])
                logger.info(f"找到腕部图像数据: {key}, shape: {wrist_image_data.shape}")
                break
        
        # 如果只找到一种图像，则复制使用
        if front_image_data is not None and wrist_image_data is None:
            wrist_image_data = front_image_data
            logger.info("未找到腕部图像，使用前视图像作为腕部图像")
        elif front_image_data is None and wrist_image_data is not None:
            front_image_data = wrist_image_data
            logger.info("未找到前视图像，使用腕部图像作为前视图像")
        elif front_image_data is None and wrist_image_data is None:
            logger.warning("未找到任何图像数据，将创建空图像")
            front_image_data = np.zeros((len(state_data), *self.image_size, 3), dtype=np.uint8)
            wrist_image_data = np.zeros((len(state_data), *self.image_size, 3), dtype=np.uint8)
        
        # 确保所有数组长度一致
        num_frames = min(len(state_data), len(action_data), len(front_image_data), len(wrist_image_data))
        
        # 处理每一帧
        for i in range(num_frames):
            # 处理前视图像
            if front_image_data.ndim == 4:  # 有时间维度
                front_img = cv2.resize(front_image_data[i], (self.image_size[1], self.image_size[0]))
            else:  # 没有时间维度，使用第一张图
                if len(front_image_data) > 0:
                    front_img = cv2.resize(front_image_data[0],
                                   (self.image_size[1], self.image_size[0]))
                else:
                    front_img = np.zeros((*self.image_size, 3), dtype=np.uint8)
            
            # 处理腕部图像
            if wrist_image_data.ndim == 4:  # 有时间维度
                wrist_img = cv2.resize(wrist_image_data[i], (self.image_size[1], self.image_size[0]))
            else:  # 没有时间维度，使用第一张图
                if len(wrist_image_data) > 0:
                    wrist_img = cv2.resize(wrist_image_data[0],
                                   (self.image_size[1], self.image_size[0]))
                else:
                    wrist_img = np.zeros((*self.image_size, 3), dtype=np.uint8)
            
            # 修复图像旋转问题：翻转180度
            front_img = cv2.flip(front_img, -1)  # -1表示180度翻转
            wrist_img = cv2.flip(wrist_img, -1)  # -1表示180度翻转
            
            frame_data = {
                "action": action_data[i].astype(np.float32),
                "observation.state": state_data[i].astype(np.float32),
                "observation.images.front": front_img,
                "observation.images.wrist": wrist_img,
                "task": task_str,
            }
            episode_frames.append(frame_data)
        
        return [episode_frames]  # 返回单个episode的列表


class RLDSProcessor:
    """RLDS数据处理器"""
    
    def __init__(self):
        if not HAS_TF:
            raise ImportError("tensorflow_datasets是RLDS处理所必需的，请运行: pip install tensorflow tensorflow_datasets")
        # 添加与HDF5Processor兼容的属性
        self.image_size = (256, 256)
        self.use_videos = False
    
    def get_default_features(self, use_videos: bool = True) -> Dict[str, Dict[str, Any]]:
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
    
    def process_dataset(self, dataset: LeRobotDataset, data_source: Union[str, Path]):
        """处理RLDS数据集"""
        # Libero数据集名称列表。根据自己需求修改
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
                # 加载RLDS数据集
                raw_dataset = tfds.load(
                    raw_dataset_name, 
                    data_dir=data_source, 
                    split="train",
                    try_gcs=False
                )
                
                for episode in raw_dataset:
                    logger.info(f"处理episode {episode_idx + 1}")
                    
                    # 获取任务描述
                    steps_list = list(episode["steps"].as_numpy_iterator())
                    task_str = f"episode_{episode_idx}"
                    
                    if steps_list and "language_instruction" in steps_list[0]:
                        task_str = steps_list[0]["language_instruction"].decode()
                    
                    # 处理episode中的每个step
                    for step_idx, step in enumerate(steps_list):
                        frame_data = {
                            "observation.images.front": step["observation"]["image"],
                            "observation.images.wrist": step["observation"]["wrist_image"],
                            "observation.state": step["observation"]["state"].astype(np.float32),
                            "action": step["action"].astype(np.float32),
                            "task": task_str,
                        }
                        # 修复API调用
                        dataset.add_frame(frame_data)
                    
                    dataset.save_episode()
                    episode_idx += 1
                    
            except Exception as e:
                logger.warning(f"处理数据集 {raw_dataset_name} 时出错: {e}")
                continue


class UnifiedConverter:
    """统一转换器类"""
    
    def __init__(self, num_workers: int = 4):
        """
        初始化统一转换器
        
        Args:
            num_workers: 并行处理的工作线程数
        """
        self.num_workers = num_workers
        self.detector = DatasetFormatDetector()
        
        # 设置多进程启动方法，提高Python 3.14兼容性
        try:
            if sys.version_info >= (3, 14) and num_workers > 1:
                logger.warning("Python 3.14+检测到，建议使用--num-workers 1以避免pickle兼容性问题")
            # 尝试设置更安全的启动方法
            if hasattr(multiprocessing, 'get_start_method'):
                current_method = multiprocessing.get_start_method()
                if current_method != 'spawn':
                    try:
                        multiprocessing.set_start_method('spawn', force=True)
                        logger.info(f"多进程启动方法从 {current_method} 更改为 spawn")
                    except RuntimeError:
                        # 如果已经设置过，无法再次设置
                        logger.debug(f"无法更改多进程启动方法，使用当前方法: {current_method}")
        except Exception as e:
            logger.warning(f"设置多进程启动方法时出错: {e}")
    
    def convert_dataset(
        self,
        data_dir: Union[str, Path],
        repo_id: str,
        output_dir: Optional[Union[str, Path]] = None,
        push_to_hub: bool = False,
        use_videos: bool = True,
        robot_type: str = "panda",
        fps: int = 20,
        task_name: str = "default_task",  # 已弃用，现在使用TASK_NAME常量
        hub_config: Optional[Dict[str, Any]] = None,
        clean_existing: bool = True,
        image_writer_threads: int = 10,
        image_writer_processes: int = 5,
        run_compute_stats: bool = False,
        **kwargs
    ) -> LeRobotDataset:
        """
        统一转换接口
        
        Args:
            data_dir: 数据目录路径
            repo_id: 数据集仓库ID
            output_dir: 输出目录
            push_to_hub: 是否推送到Hub
            use_videos: 是否使用视频格式
            robot_type: 机器人类型
            fps: 帧率
            task_name: 任务名称（已弃用，现在使用TASK_NAME常量）
            hub_config: Hub配置
            clean_existing: 是否清理现有数据集
            image_writer_threads: 图像写入线程数
            image_writer_processes: 图像写入进程数
            run_compute_stats: 是否计算统计信息
            
        Returns:
            LeRobotDataset: 转换后的数据集
        """
        data_path = Path(data_dir)
        
        # 自动检测格式
        format_type = self.detector.detect_format(data_path)
        logger.info(f"检测到数据格式: {format_type}")
        
        # 根据格式选择处理器和特征
        if format_type == "hdf5":
            processor = HDF5Processor()
            features = processor.get_default_features(use_videos)

        else:  # rlds
            processor = RLDSProcessor()
            features = processor.get_default_features(use_videos)
        
        # 设置输出路径
        if output_dir is None:
            # 如果没有指定输出目录，使用默认路径
            lerobot_root = Path("./outputs")
        else:
            lerobot_root = Path(output_dir)
        
        os.environ["HF_LEROBOT_HOME"] = str(lerobot_root)
        lerobot_dataset_dir = lerobot_root / repo_id
        
        # 清理现有数据集
        if clean_existing and lerobot_dataset_dir.exists():
            logger.info(f"清理现有数据集: {lerobot_dataset_dir}")
            shutil.rmtree(lerobot_dataset_dir)
        
        lerobot_root.mkdir(parents=True, exist_ok=True)
        
        # 创建LeRobot数据集
        logger.info(f"创建LeRobot数据集: {repo_id}")
        logger.info(f"机器人类型: {robot_type}, 帧率: {fps}, 使用视频: {use_videos}")
        logger.info(f"图像写入进程数: {image_writer_processes}, 线程数: {image_writer_threads}")
        logger.info(f"特征配置: {features.keys()}")

        # 当使用视频时，使用较小的批处理大小以平衡性能和安全性
        batch_encoding_size = 10 if use_videos else 100
        logger.info(f"批处理编码大小: {batch_encoding_size} (使用较小的批处理以防止内存问题)")

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type=robot_type,
            fps=fps,
            features=features,
            use_videos=use_videos,
            image_writer_processes=image_writer_processes,
            image_writer_threads=image_writer_threads,
            batch_encoding_size=batch_encoding_size,
        )
        
        # 处理数据
        if format_type == "hdf5":
            self._process_hdf5_data(processor, dataset, data_path, TASK_NAME)
        else:  # rlds
            # 确保是RLDSProcessor实例
            if isinstance(processor, RLDSProcessor):
                processor.process_dataset(dataset, data_path)
            else:
                raise TypeError("RLDS格式需要RLDSProcessor实例")
        
        # 移除consolidate调用，因为API可能已更改
        logger.info("数据集处理完成")
        
        # 推送到Hub
        if push_to_hub:
            if hub_config is None:
                hub_config = self._get_default_hub_config()
            logger.info("推送到Hugging Face Hub...")
            dataset.push_to_hub(**hub_config)
        
        logger.info("✅ 数据集转换完成!")
        return dataset
    
    def _process_hdf5_data(self, processor: Union[HDF5Processor, RLDSProcessor], dataset: LeRobotDataset, data_path: Path, task_name: str):
        """使用多线程处理HDF5数据"""
        # 注意：task_name参数已弃用，现在使用TASK_NAME常量
        # 确保是HDF5Processor
        if not isinstance(processor, HDF5Processor):
            raise TypeError("processor必须是HDF5Processor实例")
            
        # 查找所有episode
        episodes = []
        for ep_dir in data_path.iterdir():
            if ep_dir.is_dir():
                ep_path = ep_dir / "data" / "trajectory.hdf5"
                if ep_path.exists():
                    episodes.append(ep_path)
        
        if not episodes:
            # 直接查找HDF5文件
            episodes = list(data_path.rglob("*.hdf5")) + list(data_path.rglob("*.h5"))
        
        logger.info(f"找到 {len(episodes)} 个episode文件")
        
        # 检查Python版本和并行设置
        import sys
        use_parallel = self.num_workers > 1 and sys.version_info < (3, 14)
        
        if not use_parallel:
            # 单线程处理（Python 3.14+或num_workers=1）
            if sys.version_info >= (3, 14):
                logger.warning("检测到Python 3.14+，由于pickle兼容性问题，使用顺序处理而非并行处理")
            for ep_path in tqdm(episodes, desc="处理Episodes"):
                processor.process_episode(ep_path, dataset, TASK_NAME)
                logger.info(f"处理完成: {ep_path.name}")
        else:
            # 多线程处理
            self._process_episodes_parallel(processor, dataset, episodes, TASK_NAME)
    
    def _process_episodes_parallel(self, processor: HDF5Processor, dataset: LeRobotDataset, episodes: List[Path], task_name: str):
        """并行处理episodes，使用HDF5Processor的统一方法"""
        # 注意：task_name参数已弃用，现在使用TASK_NAME常量
        # 检查Python版本，如果是3.14+则使用顺序处理以避免pickle问题
        import sys
        if sys.version_info >= (3, 14):
            logger.warning("检测到Python 3.14+，由于pickle兼容性问题，使用顺序处理而非并行处理")
            for ep_path in tqdm(episodes, desc="处理Episodes"):
                processor.process_episode(ep_path, dataset, TASK_NAME)
                logger.info(f"处理完成: {ep_path.name}")
            return
        
        # 创建处理函数
        def process_single_episode(ep_path: Path) -> Tuple[Path, bool, List[List[Dict]]]:
            """处理单个episode并返回帧数据"""
            demos_frames = []
            try:
                with h5py.File(ep_path, "r") as file:
                    logger.debug(f"多线程处理HDF5文件键: {list(file.keys())}")
                    
                    # 使用与HDF5Processor相同的检测和处理逻辑
                    if "data" in file:
                        # 新的Libero格式：data/demo_N/...
                        demos_frames = processor._extract_libero_demo_frames(file, TASK_NAME, ep_path)
                    else:
                        # 尝试其他格式的兜底处理
                        demos_frames = processor._extract_direct_episode_frames(file, TASK_NAME, ep_path)
                    
                return ep_path, True, demos_frames
            except Exception as e:
                logger.error(f"处理 {ep_path} 失败: {e}")
                return ep_path, False, []
        
        # 并行处理
        logger.info(f"使用 {self.num_workers} 个工作线程并行处理episodes")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交所有任务
            future_to_episode = {executor.submit(process_single_episode, ep): ep for ep in episodes}
            
            # 收集结果
            for future in tqdm(concurrent.futures.as_completed(future_to_episode), total=len(episodes), desc="处理Episodes"):
                ep_path, success, demos_frames_list = future.result()
                
                if success and demos_frames_list:
                    # 每个demo作为独立的episode保存
                    for demo_idx, demo_frames in enumerate(demos_frames_list):
                        for frame_data in demo_frames:
                            # 不再移除task信息，直接使用
                            dataset.add_frame(frame_data)
                        dataset.save_episode()
                        logger.info(f"保存episode: {ep_path.name}_demo_{demo_idx}")
    
    def _get_default_hub_config(self) -> Dict[str, Any]:
        """获取默认Hub配置"""
        return {
            "tags": ["libero", "robotics", "lerobot", "unified"],
            "private": False,
            "push_videos": True,
            "license": "apache-2.0",
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Libero统一数据转换器 - 支持RLDS和HDF5格式自动识别",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 自动检测格式并转换
  python unified_converter.py \\
    --data-dir /path/to/data \\
    --repo-id username/dataset_name \\
    --push-to-hub \\
    --use-videos \\
    --num-workers 4

  # HDF5格式，指定配置文件
  python unified_converter.py \\
    --data-dir /path/to/hdf5/data \\
    --repo-id username/hdf5_dataset \\
    --config config.json \\
    --task-name "pick_and_place" \\
    --num-workers 8
        """
    )
    
    # 必需参数
    parser.add_argument("--data-dir", type=str, required=True, help="数据目录路径")
    parser.add_argument("--repo-id", type=str, required=True, help="数据集仓库ID")
    
    # 输出配置
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--push-to-hub", action="store_true", help="推送到Hub")
    parser.add_argument("--private", action="store_true", help="创建私有数据集")
    
    # 数据格式
    parser.add_argument("--use-videos", action="store_true", help="使用视频格式")
    parser.add_argument("--robot-type", type=str, default="panda", help="机器人类型")
    parser.add_argument("--fps", type=int, default=20, help="帧率")
    
    # HDF5特定参数
    parser.add_argument("--task-name", type=str, default="default_task", help="任务名称")
    
    # 性能参数
    parser.add_argument("--num-workers", type=int, default=2, help="并行工作线程数")
    parser.add_argument("--image-writer-processes", type=int, default=5, help="图像写入进程数")
    parser.add_argument("--image-writer-threads", type=int, default=1, help="图像写入线程数")
    
    # Hub配置
    parser.add_argument("--license", type=str, default="apache-2.0", help="数据集许可证")
    parser.add_argument("--tags", nargs="+", default=["libero", "robotics", "lerobot"], help="数据集标签")
    
    # 调试选项
    parser.add_argument("--verbose", action="store_true", help="详细日志")
    parser.add_argument("--dry-run", action="store_true", help="试运行模式")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 验证参数
    if not Path(args.data_dir).exists():
        logger.error(f"数据目录不存在: {args.data_dir}")
        return 1
    
    if "/" not in args.repo_id:
        logger.error(f"repo_id格式错误: {args.repo_id}")
        return 1
    
    logger.info("📋 转换配置:")
    logger.info(f"  数据源: {args.data_dir}")
    logger.info(f"  仓库ID: {args.repo_id}")
    logger.info(f"  并行线程数: {args.num_workers}")
    logger.info(f"  使用视频: {args.use_videos}")
    logger.info(f"  推送到Hub: {args.push_to_hub}")
    
    if args.dry_run:
        logger.info("✅ 试运行完成，参数验证通过")
        return 0
    
    # 执行转换
    try:
        converter = UnifiedConverter(num_workers=args.num_workers)
        
        hub_config = {
            "tags": args.tags,
            "private": args.private,
            "license": args.license,
        }

        
        dataset = converter.convert_dataset(
            data_dir=args.data_dir,
            repo_id=args.repo_id,
            output_dir=args.output_dir,
            push_to_hub=args.push_to_hub,
            use_videos=args.use_videos,
            robot_type=args.robot_type,
            fps=args.fps,
            task_name=TASK_NAME,
            hub_config=hub_config,
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=args.image_writer_threads,
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
