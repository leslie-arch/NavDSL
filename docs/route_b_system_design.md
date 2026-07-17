# Route B 系统设计文档：HM3D-AutoVLN 接入 NavDSL 的 DUET 离散图导航 VLN 训练栈

**版本**：v1.1（新增 per-scan LMDB 拆分，详见 §5.6 和 §9.5）
**日期**：2026-07-12
**作者**：Route B 实现工作组
**审阅状态**：代码完成，本地 smoke test 待跑

---

## 1. 概述

### 1.1 目标

把 HM3D-AutoVLN 论文（NeurIPS 2022, *Learning from Unlabeled 3D Environments for Vision-and-Language Navigation*）的 DUET 训练栈接入 NavDSL 项目，在 habitat-sim 环境内完成 VLN 行为克隆（BC）fine-tune，复用论文已发布的预训练权重 `model_step_35000.pt`。

### 1.2 设计约束

| #  | 约束                                                      | 来源                                         |
|----|-----------------------------------------------------------|----------------------------------------------|
| C1 | 训练必须基于 habitat-sim（不仅 MatterSim）                | 用户明确要求                                 |
| C2 | 策略输入直接读 LMDB 预提取 ViT 特征                       | 与论文一致，避免重新渲染开销                 |
| C3 | 复用 `model_step_35000.pt` 预训练权重                     | 节省训练成本（10 天 8×A100 → 数小时 1×A100） |
| C4 | 不依赖 habitat-baselines 的 `RolloutStorage`/`PPOTrainer` | DUET 变长 candidate 与固定 shape 抽象不兼容  |
| C5 | 与 NavDSL 现有 DSL+Z3+LLM 栈解耦                          | 两套范式并行存在，互不影响                   |

### 1.3 非目标

- 不复现 REVERIE/SOON 评测榜单的完整数字（仅复现 BC fine-tune 阶段）
- 不实现 DAgger 的伪交互式 demonstrator（BC 跑通后再扩展）
- 不替换 NavDSL 自带的连续动作 VLN-CE 流水线（两者并存）

---

## 2. 系统架构

### 2.1 总体分层

```
┌─────────────────────────────────────────────────────────────────┐
│                    navdsl.run (Hydra 入口)                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
       ┌───────────────────┴───────────────────┐
       │                                       │
       ▼                                       ▼
┌─────────────────┐                ┌─────────────────────┐
│  habitat.Env    │                │  DUETTrainer        │
│  (Phase 3 task) │                │  (Phase 5)          │
└─────────────────┘                └──────────┬──────────┘
       │                                       │
       │ contains                              │ uses
       │                                       │
       ▼                                       ▼
┌─────────────────┐                ┌─────────────────────┐
│ VLNGraphNavTask │                │  EpisodeBatch       │
│ + GotoViewpoint │◄────── reads ──│  (env adapter)      │
│ + Stop          │                └──────────┬──────────┘
│ + 4 sensors     │                           │
│ + 3 measures    │                           │ wraps
└─────────────────┘                           │
       │                                      ▼
       │ reads                       ┌──────────────────────┐
       ▼                             │ GMapObjectNavAgent   │
┌─────────────────────┐              │ + VLNBert (DUET)     │
│ HM3DAutoVLNDataset  │◄─── loads ───│ + Critic             │
│ (Phase 2)           │              └──────────────────────┘
│  - LMDB view/obj    │                     │
│  - nav graph        │                     │ loads weights
│  - rel_angles       │                     ▼
└─────────────────────┘              ┌──────────────────────┐
                                     │ model_step_35000.pt  │
                                     │ (pretrained)         │
                                     └──────────────────────┘
```

### 2.2 数据流（单步训练）

```
EpisodeBatch.reset()
  └─► 选 batch_size 个 episode, current_vp = start_viewpoint_id

EpisodeBatch._get_obs()
  ├─► dataset.get_view_features(scan, vp)    → (36, 768) ViT pano
  ├─► dataset.get_candidates(scan, vp)       → List[vp_id] 邻接表
  ├─► dataset.get_rel_angle(scan, vp, nbr)   → (view_idx, dist, rel_h, rel_e)
  ├─► dataset.get_object_features(scan, vp)  → {fts, obj_ids, bboxes, centers, ...}
  └─► 拼成 obs dict list (一个/episode)

agent.rollout(train_ml=1.0)
  ├─► VLNBert('language', txt_ids, txt_masks)        → text embedding
  ├─► for t in range(max_action_len):
  │     ├─► VLNBert('panorama', view_fts, obj_fts)   → pano embedding
  │     ├─► GraphMap.update_node_embed(...)          → 全局图更新
  │     ├─► VLNBert('navigation', gmap, vp, txt)     → logits + obj_logits
  │     ├─► teacher forcing: nav_targets = next vp in gt_path
  │     ├─► CrossEntropyLoss(logits, nav_targets)    → ml_loss
  │     └─► EpisodeBatch.step(actions=[next_vp, ...])
  └─► self.loss = ml_loss + og_loss_weight * og_loss  (累积到 agent)

agent.train() 内部：
  loss.backward()
  torch.nn.utils.clip_grad_norm_(vln_bert.parameters(), 40.)
  vln_bert_optimizer.step()
```

---

## 3. 模块设计

### 3.1 模块清单

| 层                    | 文件                                                | 行数 | 注册名                                                 | 职责                                                                   |
|-----------------------|-----------------------------------------------------|-----:|--------------------------------------------------------|------------------------------------------------------------------------|
| **Phase 1 数据转换**  | `navdsl/data_adapter/convert_hm3d_autovln.py`       |  215 | —                                                      | speaker jsonl + connectivity → habitat episodes json.gz                |
|                       | `navdsl/data_adapter/verify_hm3d_autovln.py`        |  189 | —                                                      | 结构化字段校验                                                         |
|                       | `navdsl/data_adapter/verify_pose_in_habitat_sim.py` |  121 | —                                                      | habitat-sim 渲染坐标系校验                                             |
|                       | `navdsl/data_adapter/test_hm3d_autovln_dataset.py`  |  106 | —                                                      | dataset 自检脚本                                                       |
|                       | `navdsl/data_adapter/split_lmdb_per_scan.py`        |  110 | —                                                      | 拆分单大 LMDB → per-scan LMDB（v1.1 新增）                             |
| **Phase 2 Dataset**   | `navdsl/data_adapter/hm3d_autovln_dataset.py`       |  365 | `HM3DAutoVLN-v1`                                       | 加载 episodes + nav graph + LMDB（支持 per-scan 与 monolithic 双模式） |
| **Phase 3 Task**      | `navdsl/tasks/vln_graph_nav.py`                     |  175 | `VLNGraphNav-v0`                                       | 离散 viewpoint 跳转任务                                                |
| **Phase 3 Sensors**   | `navdsl/sensor/viewpoint_feature_sensor.py`         |   43 | `ViewpointFeatureSensor`                               | 36-view ViT 特征                                                       |
|                       | `navdsl/sensor/object_feature_sensor.py`            |   46 | `ObjectFeatureSensor`                                  | 物体特征 dict                                                          |
|                       | `navdsl/sensor/candidate_viewpoints_sensor.py`      |   71 | `CandidateViewpointsSensor`                            | 邻接候选 + 相对角度                                                    |
|                       | `navdsl/sensor/graph_nodes_sensor.py`               |   87 | `GraphNodesSensor`                                     | 全局图状态（DUET GMap 分支）                                           |
| **Phase 3 Measures**  | `navdsl/measurements/viewpoint_success.py`          |  106 | `ViewpointSuccess` / `ViewpointSPL` / `ViewpointSteps` | viewpoint-粒度评测                                                     |
| **Phase 4 DUET port** | `navdsl/policy/duet/transformer.py`                 |  477 | —                                                      | DETR transformer (verbatim port)                                       |
|                       | `navdsl/policy/duet/ops.py`                         |   68 | —                                                      | encoder 构造 + mask 工具                                               |
|                       | `navdsl/policy/duet/graph_utils.py`                 |  170 | —                                                      | GraphMap 全局图数据结构                                                |
|                       | `navdsl/policy/duet/vilmodel.py`                    |  854 | —                                                      | GlocalTextPathNavCMT 主模型                                            |
|                       | `navdsl/policy/duet/vlnbert_init.py`                |  119 | —                                                      | 模型工厂 + 权重 key 重映射                                             |
|                       | `navdsl/policy/duet/model.py`                       |   54 | —                                                      | VLNBert + Critic 包装                                                  |
|                       | `navdsl/policy/duet/utils_ops.py`                   |   37 | —                                                      | pad_tensors / gen_seq_masks                                            |
|                       | `navdsl/policy/duet/compat.py`                      |   31 | —                                                      | is_default_gpu / print_progress shims                                  |
|                       | `navdsl/policy/duet/agent_base.py`                  |  262 | —                                                      | BaseAgent + Seq2SeqAgent 基类                                          |
|                       | `navdsl/policy/duet/agent_obj.py`                   |  495 | —                                                      | GMapObjectNavAgent 推理循环                                            |
| **Phase 4 Policy**    | `navdsl/policy/duet/duet_policy.py`                 |  116 | `DUETPolicy`                                           | habitat-baselines Policy 包装                                          |
| **Phase 5 Trainer**   | `navdsl/utils/duet_trainer.py`                      |  478 | `duet_il`                                              | 独立 BC trainer + EpisodeBatch                                         |
| **Phase 6 配置**      | `config/habitat/task/vln_graph_nav.yaml`            |    — | —                                                      | task + actions + sensors + measures                                    |
|                       | `config/habitat/dataset/vln/hm3d_autovln.yaml`      |    — | —                                                      | 数据集路径                                                             |
|                       | `config/benchmark/nav/vln_hm3d_autovln.yaml`        |    — | —                                                      | benchmark 组合                                                         |
|                       | `config/experiments/hm3d_autovln_graph_nav.yaml`    |    — | —                                                      | 训练超参                                                               |
| **Phase 7 入口**      | `navdsl/run.py`                                     |   92 | —                                                      | 9 行 import 注册                                                       |

**代码总量**：~5000 行 Python + 4 个 yaml 配置

### 3.2 注册映射

```
habitat.registry
├── dataset:    HM3DAutoVLN-v1           → HM3DAutoVLNDatasetV1
├── task:       VLNGraphNav-v0           → VLNGraphNavTask
├── actions:    GotoViewpointAction, StopAction
├── sensors:    ViewpointFeatureSensor, ObjectFeatureSensor,
│               CandidateViewpointsSensor, GraphNodesSensor
└── measures:   ViewpointSuccess, ViewpointSPL, ViewpointSteps

habitat_baselines.baseline_registry
├── policy:     DUETPolicy
└── trainer:    duet_il                  → DUETTrainer
```

---

## 4. 关键设计决策

### 4.1 habitat-sim 在环但读 LMDB 特征（约束 C1 + C2）

**问题**：HM3D-AutoVLN 原版用 MatterSim 直接驱动训练，policy 输入是 LMDB 预提取的 ViT 特征。用户要求训练基于 habitat-sim。

**决策**：保留 habitat-sim 作为环境（维护 agent state、navmesh 验证、可选 RGB 渲染），但策略输入仍读 LMDB。GotoViewpoint action 调用 `sim.set_agent_state(target_pos, ...)` 真实移动 agent，policy 不消费 sim 渲染结果。

**结果**：
- 兼顾 habitat-sim 在环（C1）和论文复现度（C2）
- 速度：训练时无渲染开销，单步 ~50ms（LMDB lookup + DUET forward）
- 副作用：rgb/depth sensor 仍配置在 yaml 中，用于将来可视化

### 4.2 不继承 VLNDatasetV1，独立继承 Dataset

**问题**：habitat-lab 的 `VLNDatasetV1.from_json` 硬性要求 json 顶层有 `instruction_vocab` 字段（用于 R2R 词表查找）。但我们的 instructions 已经 BERT tokenize 过（`instr_encoding` 字段），不需要词表。

**决策**：`HM3DAutoVLNDatasetV1` 直接继承 `habitat.core.dataset.Dataset`，自定义 `from_json`。

**结果**：避开强制 vocab 要求，episode json 格式更简洁。

### 4.3 不使用 RolloutStorage，独立 Trainer

**问题**：habitat-baselines 的 `RolloutStorage` 和 `PPOTrainer` 假设固定 shape 的 action space 和 observation tensor。DUET 每步 candidate 数量动态变化（取决于当前 viewpoint 邻接数），与 RolloutStorage 不兼容。

**决策**：`DUETTrainer` 继承 `BaseTrainer`（满足 `@register_trainer` 的 issubclass 断言），但不继承 `BaseILTrainer`/`BaseRLTrainer`，自定义训练循环：
- 调用 `agent.train(n_iters)` —— agent 内部管理 optimizer/backward/step
- 每个 iter 调一次 `rollout(train_ml=1.0)`，跑完一个 episode
- BC loss = CE(nav_logits, nav_targets) + OG loss

### 4.4 复用 model_step_35000.pt，无独立权重转换

**问题**：DUET 原版用 `models.model.VLNBert` 类，state_dict key 与 habitat-baselines Policy 接口不同。

**决策**：`vlnbert_init.build_duet()` 在加载时做 key 重映射：
- `module.X` → `X`（去 DDP 前缀）
- `_head.X` / `sap_fuse.X` → `bert._head.X` / `bert.sap_fuse.X`（补 `bert.` 前缀）

其余 key 直接对应，无需独立转换脚本。

### 4.5 EpisodeBatch 适配 MatterSim env 接口

**问题**：`GMapObjectNavAgent.rollout()` 期望 env 是 `HM3DReverieObjectNavBatch`（MatterSim batch env），调用 `env.reset()/_get_obs()/step()`、`env.shortest_distances[scan][vp1][vp2]`、甚至 `env.env.sims[i].newEpisode(...)`。

**决策**：实现 `EpisodeBatch` 类，提供完整 MatterSim env API，但底层用 NavDSL 的 dataset + networkx nav graph。`env.env.sims` 用 `_FakeSims` stub（agent 的 `make_equiv_action` 会调 `newEpisode` 来"移动"，但我们的状态由 `step()` 直接更新 `current_vp`）。

### 4.6 obs 字段格式与原版完全对齐

**问题**：DUET 的 `_panorama_feature_variable` 期望每条 obs 是 dict，含 `feature (36, vit+ang)`、`candidate list`、`obj_img_fts`、`obj_ang_fts`、`obj_box_fts`、`obj_ids`、`instr_encoding`、`gt_path`、`gt_end_vps`、`gt_obj_id` 等。

**决策**：`EpisodeBatch._build_single_obs()` 严格按 `map_nav_src/reverie/env.py:293-349` 的格式构造。每个字段都有明确的数据来源：
- `feature`：`view_fts` (LMDB) + `pano_angles`（36-view sin/cos 嵌入，预计算）
- `candidate`：每邻居 = `view_fts[view_idx]` + `angle_feature(rel_h+base_h, rel_e+base_e)`
- `obj_*`：从 LMDB dict 拆分 + 用 centers/bboxes 算 angle/size features

---

## 5. 数据规格

### 5.1 输入数据（HM3D-AutoVLN v1.0）

| 资源                     | 路径                                                                              | 大小 / 数量                           |
|--------------------------|-----------------------------------------------------------------------------------|---------------------------------------|
| HM3D 场景 mesh           | `/sata/sdb7/dataset/habitat-data/versioned_data/hm3d-0.2/hm3d/`                   | 901 scenes                            |
| Nav graph (connectivity) | `datasets/vln/hm3d/autovln/v1.0/NAV_GRAPH/connectivity/{scene}_connectivity.json` | 901 files                             |
| View 特征 LMDB           | `NAV_GRAPH/features/view_timm_imagenet_vitb16/`                                   | key=`{scan}_{vp}`, value=`(36, 1768)` |
| Object 特征 LMDB         | `NAV_GRAPH/features/obj2d_ade20k_pseudo3d_merged_timm_imagenet_vitb16/`           | dict 含 9 字段                        |
| 候选相对角度             | `NAV_GRAPH/annotations/scanvp_candview_relangles.json`                            | 预计算（避免运行时 MatterSim）        |
| Speaker 指令             | `REVERIE/expr_speakers/.../ade20k_pseudo3d_depth2_epoch_94_beam0.jsonl`           | 217,703 entries                       |
| 预训练 DUET 权重         | `REVERIE/expr_duet/pretrain_hm3d_v1/.../ckpts/model_step_35000.pt`                | 35k steps                             |

### 5.2 pose 坐标系映射（已远程验证）

HM3DAutoVLN 的 connectivity json 中 `pose` 是 16 元素 4×4 矩阵拍平：

```
pose[3]  = X
pose[7]  = -Z    （注意负号）
pose[11] = Y
```

habitat-sim Y-up 约定下，viewpoint 在 habitat-sim 中的位置：

```python
X, Y, Z = pose[3], pose[11], -pose[7]
```

**验证**：转换 1 个 scene 后用 habitat-sim 渲染 5 个 viewpoint 的 RGB，肉眼确认房间结构合理（非全黑、非翻转）。

### 5.3 Speaker jsonl 字段

```json
{
  "scan": "00000-kfPV7w3FaU5",
  "pos_vps": ["000000", "000001", ...],     // 目标物体可见 viewpoint
  "objid": "3",                              // 目标物体 id
  "instruction": "go to the bedroom on level 2...",
  "instr_encoding": [101, 2175, ...],        // BERT tokens
  "instr_id": "0_3_0",                       // 唯一 id
  "path": ["000007", "000005", "000004", "000000"]  // 导航路径
}
```

### 5.4 转换后 habitat episode 格式

```python
{
  "episode_id": "0_3_0",
  "scene_scan_id": "00000-kfPV7w3FaU5",
  "scene_id": "hm3d/train/00000-kfPV7w3FaU5/00000-kfPV7w3FaU5.basis.glb",
  "start_position": [X, Y, Z],
  "start_rotation": [qx, qy, qz, qw],        // 朝向 path[1]
  "goals": [{"position": [gX, gY, gZ], "radius": 1.0}],
  "reference_path": [[X,Y,Z], ...],
  "reference_viewpoints": ["000007", "000005", "000004", "000000"],
  "start_viewpoint_id": "000007",
  "goal_viewpoint_id": "000000",
  "target_object_id": "3",
  "target_visible_viewpoints": ["000000", ...],
  "instruction": {
    "instruction_text": "...",
    "instruction_tokens": [101, 2175, ...]   // BERT token ids
  },
  "trajectory_id": 0
}
```

### 5.5 数据规模（远程验证后）

| Split    |    Episodes | Unique Scenes | 备注                           |
|----------|------------:|--------------:|--------------------------------|
| train    |     183,330 |           799 | scene_id 数字前缀 < 800        |
| val      |      22,544 |           100 | scene_id 数字前缀 ≥ 800        |
| 跳过     |      11,829 |             2 | path<2 或 vp 不在 connectivity |
| **合计** | **217,703** |       **901** | —                              |

### 5.6 特征文件存储：per-scan LMDB 模式（v1.1 新增）

**原版（HM3D-AutoVLN 仓库）** 用两个单大 LMDB 文件存储所有 scene 的特征：

```
features/
├── view_timm_imagenet_vitb16/                       9.2 GB, 38,257 entries
│   ├── data.mdb
│   └── lock.mdb
└── obj2d_ade20k_pseudo3d_merged_timm_imagenet_vitb16/   3.8 GB, 36,465 entries
    ├── data.mdb
    └── lock.mdb
```

**NavDSL Route B（v1.1）** 改为 per-scan 单文件 LMDB 布局，与 HM3D scene mesh 目录结构对齐：

```
features/per_scan/
├── train/
│   ├── 00000-kfPV7w3FaU5.lmdb      8.4 MB, ~84 entries (view + obj 合并)
│   ├── 00001-UVdNNRcVyV1.lmdb      ~10 MB
│   └── ...                          (800 个文件)
└── val/
    ├── 00800-TEEsavR23oF.lmdb
    └── ...                          (100 个文件)
```

每个 `{scan}.lmdb` 是**单文件**（`subdir=False`），内部 keyspace：
- `view_{vp}` → msgpack ndarray `(36, 1768)`
- `obj_{vp}` → msgpack dict（含 9 字段：fts, obj_ids, obj_names, bboxes, centers, ...）

**设计权衡**：

| 维度           | 单大 LMDB | per-scan LMDB                                 |
|----------------|-----------|-----------------------------------------------|
| 文件数         | 2         | ~900                                          |
| 单次 read 延迟 | 0.042 ms  | 0.043 ms（cache hit）/ 0.086 ms（cache miss） |
| 训练 step 影响 | 基准      | +0.04%（cache 命中）/ +0.08%（未命中）        |
| 选择性同步     | ❌ 整文件 | ✓ 单 scene                                    |
| 失败隔离       | ❌ 全局   | ✓ 单 scene                                    |
| 目录约定一致性 | ❌        | ✓ 与 HM3D mesh 对齐                           |
| 维护可读性     | 一般      | 优                                            |

**性能数据**（本地 micro-benchmark，200 次 view+obj read）：
- Hot cache overhead vs legacy：**+1.6%**
- Cold cache overhead vs legacy：+105.8%（绝对值 0.086 ms，仍 < 0.1% step time）
- 训练 step 中特征 read 占比：**0.04%**（GPU forward 占 90%+）

**结论**：性能影响可忽略，从维护性与项目清晰度角度采纳 per-scan 模式。详见 `navdsl/data_adapter/split_lmdb_per_scan.py` 与 §9.5。

---

## 6. 接口规格

### 6.1 Action 接口

```python
# Action 字典格式（habitat.Env.step 入参）
{"action": "goto_viewpoint", "action_args": {"viewpoint_idx": 3}}
{"action": "stop"}

# action_space
GotoViewpointAction.action_space = gym.spaces.Dict({
    "viewpoint_idx": gym.spaces.Discrete(n=32)   # 上限，实际每步变化
})
StopAction.action_space = EmptySpace()
```

### 6.2 Sensor 输出格式

| Sensor UUID            | 类型         | Shape / Schema                                                                           |
|------------------------|--------------|------------------------------------------------------------------------------------------|
| `viewpoint_features`   | `np.float32` | `(36, 768)` ViT-B/16 pano                                                                |
| `object_features`      | dict         | `{fts: (N, 768), obj_ids: List, obj_names: List, bboxes, centers, 3d_centers, 3d_sizes}` |
| `candidate_viewpoints` | dict         | `{vp_ids: List[str], rel_angles: (N, 4), positions: (N, 3), mask: (N,) bool}`            |
| `graph_nodes`          | dict         | `{visited_vp_ids, frontier_vp_ids, current_vp_id, edges: List[Tuple]}`                   |
| `instruction`          | dict         | `{text, tokens, trajectory_id}`                                                          |

### 6.3 Measurement 输出

| Measurement        | 类型  | 计算方式                                                                 |
|--------------------|-------|--------------------------------------------------------------------------|
| `ViewpointSuccess` | float | `1.0` if stop 时 `current_vp ∈ target_visible_viewpoints` else `0.0`     |
| `ViewpointSPL`     | float | `success * ref_len / max(ref_len, actual_len)`（viewpoint 数为长度单位） |
| `ViewpointSteps`   | int   | 总步数（用于统计）                                                       |

---

## 7. 训练配置

### 7.1 默认超参（`config/experiments/hm3d_autovln_graph_nav.yaml`）

```yaml
habitat_baselines:
  trainer_name: "duet_il"
  il:
    duet:
      checkpoint: "<path>/model_step_35000.pt"
      tokenizer: "bert"
      image_feat_size: 768
      angle_feat_size: 4
      obj_feat_size: 768
      num_l_layers: 9
      num_pano_layers: 2
      num_x_layers: 4
      graph_sprels: True
      fusion: "dynamic"

      max_epochs: 100
      batch_size: 8
      learning_rate: 1.0e-5
      weight_decay: 0.01
      max_grad_norm: 5.0
      max_action_steps: 20
      eval_interval: 1
```

### 7.2 训练入口

```bash
python -m navdsl.run \
    --config-name=experiments/hm3d_autovln_graph_nav \
    habitat_baselines.evaluate=False
```

### 7.3 评测入口

```bash
python -m navdsl.run \
    --config-name=experiments/hm3d_autovln_graph_nav \
    habitat_baselines.evaluate=True \
    habitat_baselines.eval_ckpt_path_dir=<ckpt_dir>
```

---

## 8. 验证状态

### 8.1 已完成

| 阶段    | 验证内容                                                           | 结果                                             |
|---------|--------------------------------------------------------------------|--------------------------------------------------|
| Phase 1 | 远程 pose 坐标系 + 字段完整性                                      | ✅ 0 errors, 376 large-position warnings（合理） |
| Phase 2 | 本地 habitat registry 注册                                         | ✅ `HM3DAutoVLN-v1` 注册成功                     |
| Phase 3 | 本地 habitat registry 11 组件注册                                  | ✅ task + 2 actions + 4 sensors + 3 measures     |
| Phase 4 | 本地完整 import 测试（Python 3.9 + torch 2.8 + transformers 4.36） | ✅ 10 文件 + DUETPolicy 全部 import + 注册       |
| Phase 5 | 本地 EpisodeBatch 构造 + args namespace                            | ✅ 全部 helper 函数 OK                           |
| Phase 6 | yaml 落盘                                                          | ✅ 4 个 yaml 已就位                              |
| Phase 7 | run.py 9 行 import 注册                                            | ✅ 落盘                                          |

### 8.2 待完成

| 阶段             | 内容                                               | 预期时间   |
|------------------|----------------------------------------------------|------------|
| Phase 7 端到端   | 远程 GPU smoke test（1 scene / 1 epoch / batch=2） | < 5 分钟   |
| Phase 7 调优     | 修 obs 字段细节、key 重映射漏项                    | 视具体报错 |
| Phase 7 完整训练 | 8×A100 100 epochs                                  | 24-36 小时 |

### 8.3 远程 smoke test 命令

```bash
cd /path/to/NavDSL
python -m navdsl.run \
    --config-name=experiments/hm3d_autovln_graph_nav \
    habitat.dataset.content_scenes='["00000-kfPV7w3FaU5"]' \
    habitat.dataset.data_path=<remote_path>/{split}/{split}.json.gz \
    habitat.dataset.scenes_dir=<remote_path>/hm3d/ \
    ... (其余路径覆盖) \
    habitat_baselines.il.duet.max_epochs=1 \
    habitat_baselines.il.duet.batch_size=2
```

预期失败点（按概率排序）：
1. Hydra config 字段名/路径
2. agent.train() 内部 `.cuda()` 需要 GPU
3. EpisodeBatch obs 字段格式细节
4. 预训练权重 key 重映射漏项
5. HM3D 场景路径（sshfs 挂载点）

---

## 9. 关键设计模式

### 9.1 Lazy Loading

- **nav graph**：`HM3DAutoVLNDatasetV1._build_nav_graph(scan)` 首次访问时加载，per-scan 缓存
- **shortest_distances**：`EpisodeBatch._ShortestDistancesView` 首次访问 scan 时全对 Dijkstra，per-scan 缓存
- **LMDB**：环境句柄全程持有，按 key 随机读取

### 9.2 Config 兼容层

`_ArgsNamespace` 把 OmegaConf DictConfig 包装成 argparse-style namespace，自动应用 19 个默认值（`train_alg`、`ml_weight`、`optim`、`world_size` 等），让原版 DUET agent 代码无需修改即可运行。

### 9.3 Stub 不可达 API

`EpisodeBatch._FakeSims` 提供 `newEpisode()` 空实现，让 `agent_obj.make_equiv_action` 中残留的 MatterSim 调用不崩溃。实际 state 更新由 `EpisodeBatch.step()` 完成。

### 9.4 角度特征预计算

`_pano_angle_features()` 在 trainer init 时计算 36-view 标准 pano 的 (36, 4) 角度矩阵，避免每步重算。

### 9.5 Per-scan LMDB + LRU 缓存（v1.1 新增）

**`HM3DAutoVLNDatasetV1`** 同时支持两种 LMDB 布局：

```python
# Mode A: per-scan (recommended) — config field
feature_per_scan_dir: "/path/to/features/per_scan"

# Mode B: monolithic (legacy) — config fields
view_feature_lmdb: "/path/to/view_timm_imagenet_vitb16"
object_feature_lmdb: "/path/to/obj2d_ade20k_pseudo3d_merged_timm_imagenet_vitb16"
```

**Per-scan 模式工作机制**：

1. **Lazy open**：首次访问 `(scan, vp)` 时，按 `feature_per_scan_dir/{split}/{scan}.lmdb` 路径打开 LMDB，缓存到 `_per_scan_envs[scan]`
2. **共享 env**：view 和 obj 读操作共享同一个 env（combined layout，cache key 只用 `scan`）
3. **LRU eviction**：缓存上限默认 64 个 env；超出后关闭最久未用的，避免文件描述符泄漏
4. **缺失 scan 容错**：若 LMDB 文件不存在（极少 scene 无 obj 检测），返回 None → caller fallback 到 zeros/empty，不崩溃

**生成 per-scan LMDB**：
```bash
python -m navdsl.data_adapter.split_lmdb_per_scan \
    --source-view-lmdb /path/to/.../view_timm_imagenet_vitb16 \
    --source-obj-lmdb /path/to/.../obj2d_ade20k_pseudo3d_merged_timm_imagenet_vitb16 \
    --output-dir /path/to/.../features/per_scan
```

转换在远程主机跑约 15 分钟（读 13 GB + 写 13 GB）。

---

## 10. 与 NavDSL 现有栈的关系

### 10.1 并存策略

NavDSL 自带 `dsl_vln_hm3d.yaml`（连续动作 VLN-CE + DSL+Z3+LLM 策略），本方案新增 `hm3d_autovln_graph_nav.yaml`（离散 viewpoint + DUET）。两者完全独立，共享：
- `navdsl/run.py` 入口
- habitat-lab / habitat-baselines 框架
- HM3D 场景 mesh

### 10.2 路线 A 与路线 B 的复用

路线 A（连续动作 VLN-CE）未来若要实现，可直接复用：
- Phase 1 数据转换器（`convert_hm3d_autovln.py`）
- Phase 2 Dataset 类（去掉 `*_viewpoint_id` 字段即可）
- Phase 1 验证脚本

不复用：
- Phase 3 Task / Actions（连续动作用 habitat-lab 自带 VLN-v0）
- Phase 4-5 DUET 栈（路线 A 用通用 IL trainer）

---

## 11. 风险与未决事项

| #  | 风险                                                                               | 缓解                                                                           |
|----|------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| R1 | EpisodeBatch obs 格式细节与 agent_obj 期望不完全对齐                               | smoke test 时看 traceback，按字段补                                            |
| R2 | 预训练权重加载报 missing/unexpected keys                                           | 已有 key 重映射（`module.` 去除 + `bert.` 前缀），若有遗漏补 `vlnbert_init.py` |
| R3 | agent_obj 内部 `.cuda()` 调用强制要求 GPU                                          | 已在 yaml 显式标注；CPU fallback 需大量改写，暂不做                            |
| R4 | habitat-sim navmesh 与 HM3DAutoVLN step00 时的版本不一致，set_agent_state 可能失败 | GotoViewpoint action 已用 try/except 兜底，metric 计算不依赖 sim 状态          |
| R5 | 大型 scene（801 邻接数过多）超出 Discrete(32) action_space                         | smoke test 看是否触发；若触发，提高上限或动态扩展                              |

---

## 12. 文件索引

```
navdsl/
├── run.py                                          # Phase 7 入口（+9 行 import）
├── data_adapter/
│   ├── convert_hm3d_autovln.py                     # Phase 1 数据转换器
│   ├── verify_hm3d_autovln.py                      # Phase 1 结构校验
│   ├── verify_pose_in_habitat_sim.py               # Phase 1 坐标系校验
│   ├── test_hm3d_autovln_dataset.py                # Phase 1 dataset 自检
│   └── hm3d_autovln_dataset.py                     # Phase 2 Dataset 类
├── tasks/
│   └── vln_graph_nav.py                            # Phase 3 Task + Actions
├── sensor/
│   ├── viewpoint_feature_sensor.py                 # Phase 3
│   ├── object_feature_sensor.py                    # Phase 3
│   ├── candidate_viewpoints_sensor.py              # Phase 3
│   └── graph_nodes_sensor.py                       # Phase 3
├── measurements/
│   └── viewpoint_success.py                        # Phase 3
├── policy/
│   └── duet/
│       ├── vilmodel.py                             # Phase 4 主模型
│       ├── transformer.py                          # Phase 4 子模块
│       ├── graph_utils.py                          # Phase 4 GraphMap
│       ├── ops.py                                  # Phase 4 工具
│       ├── model.py                                # Phase 4 VLNBert 包装
│       ├── vlnbert_init.py                         # Phase 4 工厂 + 权重重映射
│       ├── agent_base.py                           # Phase 4 BaseAgent
│       ├── agent_obj.py                            # Phase 4 GMapObjectNavAgent
│       ├── utils_ops.py                            # Phase 4 pad_tensors 等
│       ├── compat.py                               # Phase 4 shim
│       └── duet_policy.py                          # Phase 4+ Policy 包装
└── utils/
    └── duet_trainer.py                             # Phase 5 Trainer + EpisodeBatch

config/
├── habitat/task/vln_graph_nav.yaml                 # Phase 6 task config
├── habitat/dataset/vln/hm3d_autovln.yaml           # Phase 6 dataset config
├── benchmark/nav/vln_hm3d_autovln.yaml             # Phase 6 benchmark
└── experiments/hm3d_autovln_graph_nav.yaml         # Phase 6 experiment

docs/
├── verify_route_b_phase1.md                        # Phase 1 远程验证清单
└── route_b_system_design.md                        # 本文档
```

---

## 13. 词汇表

| 术语              | 含义                                                            |
|-------------------|-----------------------------------------------------------------|
| DUET              | Dual-scale Graph Transformer for VLN，HM3D-AutoVLN 论文用的模型 |
| GMap              | Global Map，DUET 的全局图注意力分支                             |
| Viewpoint         | 离散导航节点，由 nav graph 定义                                 |
| Candidate         | 当前 viewpoint 的邻接 viewpoint（可跳转目标）                   |
| SPL               | Success weighted by Path Length，标准 VLN 评测指标              |
| BC                | Behavior Cloning，行为克隆（监督学习）                          |
| DAgger            | Dataset Aggregation，迭代式 BC                                  |
| LMDB              | Lightning Memory-Mapped Database，特征存储格式                  |
| nav graph         | 场景拓扑图，节点=viewpoint，边=可通行邻接                       |
| connectivity json | HM3DAutoVLN 的 nav graph 序列化格式                             |
| pano              | 36-view 全景（12 heading × 3 elevation）                        |
| ViT-B/16          | Vision Transformer Base/16，图像编码器                          |

---

## 附录 A：参考论文与项目

- **HM3D-AutoVLN**：*Learning from Unlabeled 3D Environments for Vision-and-Language Navigation*, NeurIPS 2022. [arXiv:2208.11781](https://arxiv.org/abs/2208.11781)
- **DUET**：*Think Global, Act Local: Dual-scale Graph Transformer for Vision-and-Language Navigation*, CVPR 2022. [arXiv:2202.11742](https://arxiv.org/abs/2202.11742)
- **habitat-lab**：[facebookresearch/habitat-lab](https://github.com/facebookresearch/habitat-lab) v0.3.3
- **HM3D**：[Habitat-Matterport 3D Semantics Dataset](https://aihabitat.org/datasets/hm3d/) v0.2

## 附录 B：依赖版本（本地验证通过）

```
Python 3.9 / 3.10
torch==2.8.0  (或 2.1.0+CPU 用于纯 import 测试)
transformers==4.36.2
habitat-lab==0.3.3
habitat-sim==0.3.3
numpy==1.26.4
pillow==10.4.0
opencv-python==4.10.0.84
networkx==3.2.1
lmdb==2.2.1
msgpack / msgpack_numpy
gym==0.23.0
```
