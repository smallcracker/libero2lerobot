# Libero HDF5/RLDS 转 LeRobot v2.1

本项目使用 `libero_rlds_converter.py` 将 Libero HDF5 或 RLDS 数据转换为
LeRobot v2.1（legacy layout）格式。

当前真正的命令行入口是：

```bash
python libero_rlds_converter.py
```

`main.py` 目前只是占位程序，README 中不要使用 `run_converter.py` 或
`main.py` 作为转换入口。

## 支持范围

- 自动检测 HDF5 和 RLDS。检测到两种格式时优先使用 HDF5。
- HDF5 转换为单线程执行；每个 `data/demo_N` 会被写成一个 LeRobot episode。
- 输出 LeRobot v2.1 目录结构，包括 parquet、可选视频和 `meta` 元数据。
- HDF5 图像会缩放为 `256 x 256 x 3`。
- 使用 `--use-videos` 时，图像写入 `videos/` 下的 MP4 文件；否则图像数组直接写入 parquet。

## 安装

项目要求 Python 3.10 或更高版本。使用 uv 时：

```bash
uv sync
uv pip install --python .venv pandas pyarrow jsonlines
```

或使用 pip：

```bash
pip install -e . pandas pyarrow jsonlines
```

`libero_rlds_converter.py` 直接导入 `pandas`、`pyarrow` 和 `jsonlines`；如果
当前环境没有这些包，需要显式安装。

RLDS 支持是可选的：

```bash
pip install tensorflow tensorflow-datasets
```

没有安装 `tensorflow-datasets` 时，HDF5 转换仍可使用，但 RLDS 转换不可用。

## 基本用法

### HDF5 转换

```bash
python libero_rlds_converter.py \
  --data-dir /path/to/hdf5_data \
  --output-dir /path/to/output_root \
  --repo-id local/libero_dataset \
  --use-videos \
  --robot-type panda \
  --fps 20 \
  --verbose
```

输出路径由 `output-dir` 和 `repo-id` 拼接得到：

```text
/path/to/output_root/local/libero_dataset
```

如果省略 `--output-dir`，默认输出到：

```text
~/.cache/huggingface/lerobot/<repo-id>
```

`repo-id` 必须包含 `/`，例如 `stack-cube-fix/franka`。它在本地路径中是
子目录名，不代表脚本会自动上传到 Hugging Face Hub。

### 当前数据集示例

将 `/data/yangnan/VLA/dataset/stack-cube-rotate` 转换到
`/data/yangnan/VLA/dataset/stack-cube-fix/franka`：

```bash
python libero_rlds_converter.py \
  --data-dir /data/yangnan/VLA/dataset/stack-cube-rotate \
  --output-dir /data/yangnan/VLA/dataset \
  --repo-id stack-cube-fix/franka \
  --use-videos \
  --robot-type panda \
  --fps 20 \
  --clean-existing \
  --verbose
```

`--clean-existing` 会删除已经存在的
`/data/yangnan/VLA/dataset/stack-cube-fix/franka`，请确认目标目录可以重建后
再使用。

### RLDS 转换

```bash
python libero_rlds_converter.py \
  --data-dir /path/to/rlds_data \
  --output-dir /path/to/output_root \
  --repo-id local/libero_rlds \
  --use-videos \
  --fps 20
```

RLDS 处理器会依次尝试加载以下固定数据集名称：

- `libero_10_no_noops`
- `libero_goal_no_noops`
- `libero_object_no_noops`
- `libero_spatial_no_noops`

每个数据集需要能被 `tensorflow_datasets.load(..., data_dir=<data-dir>, split="train")`
加载，并提供 `steps`，其中包含 `observation.image`、
`observation.wrist_image`、`observation.state` 和 `action`。

## 输入 HDF5 格式

脚本会递归查找 `.hdf5` 和 `.h5` 文件。支持直接放置文件：

```text
hdf5_data/
├── dataset_1.hdf5
├── dataset_2.hdf5
└── ...
```

也支持 Libero episode 目录：

```text
hdf5_data/
├── episode_001/
│   └── data/
│       └── trajectory.hdf5
└── episode_002/
    └── data/
        └── trajectory.hdf5
```

每个 HDF5 文件必须包含 `data`，并在其中包含一个或多个按数字命名的
`demo_N` 组：

```text
file.hdf5
└── data/
    └── demo_0/
        ├── actions          [N, 7]
        └── obs/
            ├── eef_pos      [N, 3]
            ├── eef_quat     [N, 4]
            ├── gripper_pos  [N, 2]
            ├── table_cam    [N, H, W, 3]
            └── wrist_cam    [N, H, W, 3]
```

图像也可以使用以下另一组名称：

```text
obs/agentview_rgb
obs/eye_in_hand_rgb
```

其中 `eef_quat` 按 `[x, y, z, w]` 解释。脚本将
`eef_pos + 四元数转欧拉角 + gripper_pos` 拼成 8 维
`observation.state`，并将 `actions` 作为 7 维 `action`。

如果四类数据的帧数不一致，脚本会截断到最短长度。缺少 `data`、必要状态
数据集或两路图像时，该 HDF5 文件会被跳过并记录错误日志。

可以用 h5py 查看文件结构：

```bash
python - <<'PY'
import h5py
from pathlib import Path

path = Path("/path/to/file.hdf5")

with h5py.File(path, "r") as file:
    file.visititems(
        lambda name, obj: print(
            name,
            getattr(obj, "shape", ""),
            getattr(obj, "dtype", ""),
        )
    )
PY
```

## 输出格式

转换结果是一个 LeRobot v2.1 数据集：

```text
<output-dir>/<repo-id>/
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet
├── videos/                         # 仅使用 --use-videos 时生成
│   └── chunk-000/
│       ├── observation.images.front/
│       │   └── episode_000000.mp4
│       └── observation.images.wrist/
│           └── episode_000000.mp4
└── meta/
    ├── episodes.jsonl
    ├── episodes_stats.jsonl
    ├── info.json
    ├── stats.json
    └── tasks.jsonl
```

HDF5 的默认特征为：

| 特征 | 类型 | 形状 |
| --- | --- | --- |
| `observation.images.front` | `video` 或 `image` | `256 x 256 x 3` |
| `observation.images.wrist` | `video` 或 `image` | `256 x 256 x 3` |
| `observation.state` | `float32` | `8` |
| `action` | `float32` | `7` |

## 命令行参数

实际参数以 `python libero_rlds_converter.py --help` 为准：

| 参数 | 说明 |
| --- | --- |
| `--data-dir PATH` | 必填，输入数据目录 |
| `--repo-id NAME/NAME` | 必填，输出数据集相对路径，必须包含 `/` |
| `--output-dir PATH` | 输出根目录；省略时使用 `~/.cache/huggingface/lerobot/` |
| `--use-videos` | 将图像写为 MP4；不使用时图像数组写入 parquet |
| `--robot-type NAME` | 写入 `meta/info.json` 的机器人类型，默认 `panda` |
| `--fps INT` | 视频帧率、时间戳和元数据中的帧率，默认 `20` |
| `--clean-existing` | 删除同名输出目录后重新转换 |
| `--verbose` | 输出更详细的日志 |
| `--dry-run` | 只检查输入目录存在且 `repo-id` 含 `/`，不读取数据 |
| `--task-name TEXT` | 当前 CLI 接受该参数，但转换器实际使用代码中的固定任务名 |
| `--push-to-hub` | 当前 CLI 接受该参数，但脚本未实现 Hub 上传 |
| `--private` | 当前 CLI 接受该参数，但脚本未实现 Hub 权限设置 |

当前版本没有 `--config`、`--num-workers`、`--force-format`、`--tags` 或
`--image-writer-*` 参数。README、旧脚本或其他项目中的这些参数不能用于本
脚本。

HDF5 转换器当前是单线程的，不支持通过命令行设置 worker 数量。

## 故障排查

### 无法检测数据格式

确认 `--data-dir` 下递归包含 `.hdf5`/`.h5` 文件，或包含 RLDS 的
`.tfrecord*`、`dataset_info.json` 或 `features.json`。

### HDF5 文件被跳过

检查文件顶层是否有 `data`，以及每个 `data/demo_N` 是否包含：

- `actions`
- `obs/eef_pos`
- `obs/eef_quat`
- `obs/gripper_pos`
- `obs/table_cam` 和 `obs/wrist_cam`，或 `obs/agentview_rgb` 和 `obs/eye_in_hand_rgb`

### 缺少 Python 模块

```bash
pip install -e . pandas pyarrow jsonlines
```

RLDS 还需要：

```bash
pip install tensorflow tensorflow-datasets
```

### 先做参数检查

```bash
python libero_rlds_converter.py \
  --data-dir /path/to/data \
  --repo-id test/check \
  --dry-run \
  --verbose
```

注意：`--dry-run` 不会检测 HDF5 内部结构，也不会验证依赖是否完整。

## 相关脚本

- `libero_rlds_converter.py`: HDF5/RLDS 到 LeRobot v2.1 的实际转换器。
- `convert_v3_to_v2.py`: 将已有 LeRobot v3.0 数据集转换回 v2.1，与 HDF5/RLDS
  转换流程无关。
- `run.sh` 和 `debug.sh`: 包含特定机器的硬编码路径，仅供原环境参考，使用前
  需要自行修改。

## 许可证

Apache 2.0
