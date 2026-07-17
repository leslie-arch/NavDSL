# Route B Phase 1 远程验证清单

在远程主机的 conda 环境里执行。所有 `<PLACEHOLDER>` 替换为你的实际路径。

## Step 0 · 环境自检（必做）

```bash
# 0.1 激活你的 conda env
conda activate <ENV_NAME>

# 0.2 Python 版本
python --version
# 期望: Python 3.10.x

# 0.3 Phase 1 转换器最小依赖（不需要 habitat）
python -c "import json, gzip, math, os; print('stdlib ok')"
# 期望: stdlib ok

# 0.4 Phase 2+ 需要（如果还没装就先 pip install）
python -c "import habitat, habitat_baselines, habitat_sim, hydra, lmdb, torch, networkx, transformers, tqdm, PIL, numpy; print('all deps ok')"
# 期望: all deps ok
# 缺什么就装什么，例如：
#   pip install lmdb tqdm jsonlines networkx transformers
#   pip install -e /path/to/NavDSL/tools/habitat-lab/habitat-lab
#   pip install -e /path/to/NavDSL/tools/habitat-lab/habitat-baselines
```

**Step 0 失败的话**：先把依赖装齐再继续。Step 0 不通过，后续都会失败。

---

## Step 1 · 路径与代码同步确认

```bash
# 1.1 把本地变量赋值（替换为实际路径）
NAVDSL_DIR=<你的 NavDSL 路径，例如 /home/you/NavDSL>
SPEAKER_JSONL=<ade20k_pseudo3d_depth2_epoch_94_beam0.jsonl 的绝对路径>
CONNECTIVITY_DIR=<NAV_GRAPH/connectivity 的绝对路径>
HM3D_SCENES_DIR=<hm3d-0.2/hm3d 的绝对路径>
HM3D_SCENE_DATASET_CONFIG=<hm3d_annotated_basis.scene_dataset_config.json 的绝对路径>

# 1.2 验证 NavDSL 仓库可访问
ls $NAVDSL_DIR/navdsl/data_adapter/convert_hm3d_autovln.py
# 期望: 输出该文件路径，无报错

# 1.3 验证 HM3DAutoVLN 数据可访问
ls $SPEAKER_JSONL
wc -l $SPEAKER_JSONL
# 期望: 文件存在，行数 ~217703

ls $CONNECTIVITY_DIR | wc -l
# 期望: ~901

# 1.4 验证 HM3D 场景 mesh 存在
ls $HM3D_SCENES_DIR/train/00000-kfPV7w3FaU5/00000-kfPV7w3FaU5.basis.glb
ls $HM3D_SCENE_DATASET_CONFIG
# 期望: 两个文件都存在

# 1.5 创建输出目录
mkdir -p $NAVDSL_DIR/data/datasets/vln/hm3d/autovln/v1
```

**如果数据/代码还没传到远程**：先把它们 rsync 过去。例如：
```bash
# 在本地机器执行（把数据推到远程）
rsync -avzP --include='*_connectivity.json' --include='*/' --exclude='*' \
  /sata/sdb7/.../NAV_GRAPH/connectivity/ \
  remote_host:/data/hm3d_autovln/connectivity/

rsync -avzP \
  /sata/sdb7/.../REVERIE/expr_speakers/2d_gpt2_prefix4_layer2/preds/ade20k_pseudo3d_depth2_epoch_94_beam0.jsonl \
  remote_host:/data/hm3d_autovln/

# NavDSL 代码（如已 push 到 git 仓库，远程直接 clone；否则 rsync）
rsync -avzP --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
  /sata/sdb5/robotics/arch-1/NavDSL/ \
  remote_host:/home/you/NavDSL/
```

---

## Step 2 · 冒烟测试（100 条 episode，1 分钟）

```bash
cd $NAVDSL_DIR
python -m navdsl.data_adapter.convert_hm3d_autovln \
    --speaker-jsonl $SPEAKER_JSONL \
    --connectivity-dir $CONNECTIVITY_DIR \
    --output-dir /tmp/hm3d_autovln_smoke \
    --limit 100
```

**期望输出**（stderr）：
```
Loading 901 connectivity files...
100%|████████████| 901/901 [00:30<00:00, xx it/s]
Loaded 901 scenes from connectivity
Reading speaker jsonl: 217703 lines [00:05, xxxx lines/s]
Converting: 100%|██████| 100/100 [00:00<00:00, xxxx it/s]
Converted: train=100, val=0, skipped=0
Wrote 100 episodes -> /tmp/hm3d_autovln_smoke/train/train.json.gz
```

**冒烟检查**：
```bash
python -c "
import gzip, json
d = json.load(gzip.open('/tmp/hm3d_autovln_smoke/train/train.json.gz'))
ep = d['episodes'][0]
print('episode_id:', ep['episode_id'])
print('scene_scan_id:', ep['scene_scan_id'])
print('scene_id:', ep['scene_id'])
print('start_position:', ep['start_position'])
print('start_rotation:', ep['start_rotation'])
print('reference_path len:', len(ep['reference_path']))
print('reference_viewpoints:', ep['reference_viewpoints'])
print('instruction:', ep['instruction']['instruction_text'][:80])
"
```

**期望**：
- `episode_id` 是 "0_3_0" 格式
- `scene_id` 是 `hm3d/train/00000-kfPV7w3FaU5/00000-kfPV7w3FaU5.basis.glb` 格式
- `start_position` 是 3 元素 list，每个值在 [-20, 20] 区间
- `start_rotation` 是 4 元素 list（四元数），且 `qx=qz=0, qy²+qw²≈1`
- `reference_path` 至少 2 个点

---

## Step 3 · 完整转换（~5-10 分钟）

冒烟测试通过后，跑完整数据：

```bash
cd $NAVDSL_DIR
python -m navdsl.data_adapter.convert_hm3d_autovln \
    --speaker-jsonl $SPEAKER_JSONL \
    --connectivity-dir $CONNECTIVITY_DIR \
    --output-dir data/datasets/vln/hm3d/autovln/v1
```

**期望**：
```
Converted: train=~200000, val=~17000, skipped=0
Wrote ~200000 episodes -> data/.../train/train.json.gz
Wrote ~17000 episodes -> data/.../val/val.json.gz
```

（train/val 比例约 800:101，217703 × 800/901 ≈ 193k train，~24k val）

---

## Step 4 · 结构化验证（无需 GPU）

```bash
cd $NAVDSL_DIR
python -m navdsl.data_adapter.verify_hm3d_autovln \
    --episodes data/datasets/vln/hm3d/autovln/v1/train/train.json.gz \
    --speaker-jsonl $SPEAKER_JSONL \
    --connectivity-dir $CONNECTIVITY_DIR \
    --sample 3 > /tmp/verify_train.txt 2>&1
cat /tmp/verify_train.txt

python -m navdsl.data_adapter.verify_hm3d_autovln \
    --episodes data/datasets/vln/hm3d/autovln/v1/val/val.json.gz \
    --speaker-jsonl $SPEAKER_JSONL \
    --connectivity-dir $CONNECTIVITY_DIR \
    --sample 0 > /tmp/verify_val.txt 2>&1
cat /tmp/verify_val.txt
```

**期望**（关键字段）：
- `bad episodes: 0 / <N>`
- `episodes not in speaker (should be 0): 0`
- `reference_path length: min=2 median=4-5 max=~20`
- `checked N episodes, bad_vp_refs=0`
- 前 3 个 episode 的字段完整

---

## Step 5 · habitat-sim 坐标系可视化校验（GPU，最高置信度）

这是验证 `X=pose[3], Y=pose[11], Z=-pose[7]` 坐标映射正确性的决定性测试。

```bash
cd $NAVDSL_DIR
python -m navdsl.data_adapter.verify_pose_in_habitat_sim \
    --episodes data/datasets/vln/hm3d/autovln/v1/train/train.json.gz \
    --scene-dataset $HM3D_SCENE_DATASET_CONFIG \
    --scenes-dir $HM3D_SCENES_DIR \
    --output-dir /tmp/pose_check \
    --num-episodes 2 \
    --views-per-episode 5
```

**期望**：在 `/tmp/pose_check/` 下生成 ~10 张 PNG 图片。

**人工判读**：
- ✅ 图片显示合理的房间结构（地板、墙壁、家具），不是全黑/全白
- ✅ 相邻 viewpoint 的图片明显是同一房间的不同位置（不是完全无关场景）
- ❌ 图片全黑 → 相机被卡在墙里 → pose 坐标系可能反了
- ❌ 图片倒置 → rotation 四元数符号问题

把 `/tmp/pose_check/*.png` 下载到本地查看。

---

## Step 6 · 报告回传给我

把以下文件内容贴回，我据此决定是否进入 Phase 2：

1. `Step 2` 的冒烟输出（~30 行）
2. `Step 4` 的 `/tmp/verify_train.txt` 和 `/tmp/verify_val.txt` 关键行：
   - `bad episodes / total`
   - `episodes not in speaker (should be 0)`
   - `reference_path length` 行
   - `bad_vp_refs`
3. `Step 5` 的 1-2 张 PNG（如果跑了）
4. 任何 ERROR 或 traceback

---

## 常见问题排查

| 现象 | 原因 | 解决 |
|---|---|---|
| `skipped > 0` 大量跳过 | connectivity 与 speaker scan 不一致 | 检查两边 scene_id 是否对得上 |
| `start_position` 全为 0 | pose 解析错 | 复核 `parse_pose` 实现 |
| `bad_vp_refs > 0` | viewpoint id 拼接错 | 检查 connectivity 加载逻辑 |
| habitat-sim 报 `not navigable` | navmesh 不一致 | 加 `sim_cfg.allow_sliding = True` 重试 |
| habitat-sim 渲染全黑 | 相机被墙挡 / GPU 没正确初始化 | 改 `gpu_device_id` 或换 scene |
| `ModuleNotFoundError: habitat` | habitat-lab 没装 | `pip install -e tools/habitat-lab/habitat-lab` |
| `KeyError: 'basis.glb'` | HM3D 版本不对 | 确认 hm3d-0.2 不是 hm3d_v0.1 |
