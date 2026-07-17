#!/usr/bin/env python3
"""Standalone BC trainer for DUET on HM3D-AutoVLN graph-nav task.

Does NOT extend habitat-baselines' ``BaseILTrainer`` or use ``RolloutStorage``
(DUET's variable-candidate-count per step is incompatible with those fixed-shape
abstractions). The training loop mirrors
``HM3DAutoVLN/map_nav_src/reverie/main_nav_obj_hm3d.py`` but drives
the batch via :class:`EpisodeBatch` (which wraps our
:class:`HM3DAutoVLNDatasetV1`) instead of MatterSim.

Architecture:
  1. HM3DAutoVLNDatasetV1 — provides LMDB features + nav graph (Phase 2).
  2. VLNGraphNavTask — discrete viewpoint action in habitat-sim (Phase 3).
  3. VLNBert (DUET) model — ported in Phase 4.
  4. EpisodeBatch (below) — bridges the dataset to the obs format expected
     by ``GMapObjectNavAgent.rollout()``.
  5. GMapObjectNavAgent — runs the actual forward pass per step, builds
     the GMap state, computes BC loss against teacher-forced reference path.
"""
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from habitat_baselines.common.base_trainer import BaseTrainer
from habitat_baselines.common.baseline_registry import baseline_registry


# ---------------------------------------------------------------------------
# Angle helpers (mirror map_nav_src/utils/data.py:angle_feature)
# ---------------------------------------------------------------------------

def angle_feature(heading: float, elevation: float, angle_feat_size: int = 4) -> np.ndarray:
    """Sinusoidal angle embedding. Period 4 (sin/cos h, sin/cos e)."""
    return np.array(
        [math.sin(heading), math.cos(heading),
         math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
        dtype=np.float32,
    )


def _pano_angle_features(angle_feat_size: int = 4) -> np.ndarray:
    """Pre-compute (36, angle_feat_size) angle features for the standard
    12-heading x 3-elevation panorama layout (elevation in {-30, 0, 30} deg)."""
    feats = np.zeros((36, angle_feat_size), dtype=np.float32)
    for ix in range(36):
        heading = (ix % 12) * math.radians(30)
        elevation = (ix // 12 - 1) * math.radians(30)
        feats[ix] = angle_feature(heading, elevation, angle_feat_size)
    return feats


# ---------------------------------------------------------------------------
# EpisodeBatch — bridges HM3DAutoVLNDatasetV1 to GMapObjectNavAgent's env API.
# ---------------------------------------------------------------------------

class EpisodeBatch:
    """Holds a batch of episodes and exposes per-step observations in the
    format that ``GMapObjectNavAgent.rollout`` expects.

    Implements the minimal subset of the original
    ``HM3DAutoVLN/map_nav_src/reverie/env.py`` EnvBatch interface:
      * reset() / _get_obs() / step(actions)
      * shortest_distances[scan][vp1][vp2] (lazy per-scan cache)
      * env.sims[i].newEpisode(...) — faked by just updating current_vp
    """

    def __init__(
        self,
        episodes: List[Any],
        dataset,
        batch_size: int = 8,
        angle_feat_size: int = 4,
        image_feat_size: int = 768,
        obj_feat_size: int = 768,
        max_objects: Optional[int] = None,
        seed: int = 0,
    ):
        self.episodes = list(episodes)
        self.dataset = dataset
        self.batch_size = batch_size
        self.angle_feat_size = angle_feat_size
        self.image_feat_size = image_feat_size
        self.obj_feat_size = obj_feat_size
        self.max_objects = max_objects
        self._rng = random.Random(seed)

        # Standard 36-view pano angle features
        self._pano_angles = _pano_angle_features(angle_feat_size)

        # Per-scan caches
        self._shortest_distances_cache: Dict[str, Dict[str, Dict[str, float]]] = {}

        # Active batch state (set by reset)
        self.batch: List[Dict[str, Any]] = []
        self._current_vps: List[str] = []
        self._headings: List[float] = []
        self._elevations: List[float] = []
        self._ix = 0  # cursor over self.episodes

        # Fake env.sims placeholder so agent_obj.make_equiv_action's
        # `self.env.env.sims[i].newEpisode(scan, action, heading, elevation)`
        # call routes back into EpisodeBatch's state update — NavDSL doesn't
        # run a real habitat-sim per step, so we just advance the cursor.
        # The 1-element-list signature matches DUET upstream's habitat-sim API.
        batch_self = self

        class _FakeSim:
            def __init__(self, i):
                self._i = i

            def newEpisode(self, scan_list, action_list, heading_list, elevation_list):
                action = action_list[0]
                if action is None:
                    return
                batch_self._current_vps[self._i] = action
                batch_self._headings[self._i] = float(heading_list[0])
                batch_self._elevations[self._i] = float(elevation_list[0])

        class _FakeEnvEnv:
            def __init__(self, n):
                self.sims = [_FakeSim(i) for i in range(n)]

        self.env = _FakeEnvEnv(batch_size)

    # ----- batch selection -----

    def _next_minibatch(self) -> List[Dict[str, Any]]:
        """Pick the next batch_size episodes; wrap around if needed."""
        if self._ix + self.batch_size <= len(self.episodes):
            batch = self.episodes[self._ix : self._ix + self.batch_size]
            self._ix += self.batch_size
        else:
            # End of epoch: reshuffle and take from start
            self._rng.shuffle(self.episodes)
            self._ix = self.batch_size
            batch = self.episodes[: self._ix]
        return batch

    # ----- habitat-style reset / step -----

    def reset(self, **kwargs) -> List[Dict[str, Any]]:
        self.batch = self._next_minibatch()
        self._current_vps = [ep.start_viewpoint_id for ep in self.batch]
        # Heading/elevation both 0 at start — agent reads panorama anyway
        self._headings = [0.0] * len(self.batch)
        self._elevations = [0.0] * len(self.batch)
        return self._get_obs()

    def _get_obs(self) -> List[Dict[str, Any]]:
        return [self._build_single_obs(i) for i in range(len(self.batch))]

    def step(self, actions: List[Optional[str]]) -> List[Dict[str, Any]]:
        """Advance each episode's current viewpoint to ``actions[i]``.

        ``actions[i] is None`` means STOP for that episode — its viewpoint
        stays unchanged (the agent marks it as ended at a higher level).
        """
        for i, action in enumerate(actions):
            if action is None:
                continue
            self._current_vps[i] = action
            self._headings[i] = 0.0
            self._elevations[i] = 0.0
        return self._get_obs()

    # ----- obs builder -----

    def _build_single_obs(self, i: int) -> Dict[str, Any]:
        ep = self.batch[i]
        scan = ep.scene_scan_id
        vp = self._current_vps[i]
        base_heading = self._headings[i]
        base_elevation = self._elevations[i]

        # 1. View features: (36, image_feat_size) ViT
        view_fts = self.dataset.get_view_features(scan, vp)
        # 2. Build full panorama: concat(view_fts, pano_angle_features)
        # Same angle features for all 36 views (relative to current base)
        view_ang = self._pano_angles.copy()  # (36, angle_feat_size)
        # Adjust angles by base heading/elevation of the agent's current view
        view_ang[:, 0] = np.sin(np.arange(36) * (np.pi / 6) - base_heading)
        view_ang[:, 1] = np.cos(np.arange(36) * (np.pi / 6) - base_heading)
        view_ang[:, 2] = np.sin(((np.arange(36) // 12) - 1) * (np.pi / 6) - base_elevation)
        view_ang[:, 3] = np.cos(((np.arange(36) // 12) - 1) * (np.pi / 6) - base_elevation)

        feature = np.concatenate([view_fts, view_ang], axis=1)  # (36, vit+ang)

        # 3. Candidates: neighbors of current viewpoint
        candidate = self._build_candidates(scan, vp, base_heading, base_elevation)

        # 4. Object features
        obj_img_fts, obj_ang_fts, obj_box_fts, obj_ids = self._build_object_features(
            scan, vp, base_heading, base_elevation
        )

        # 5. Current viewpoint world-space position — required by
        # GMap.update_graph (graph_utils.py:107) to populate node_positions
        # and compute edge distances to candidates.
        try:
            cur_pos = tuple(self.dataset.get_viewpoint_position(scan, vp))
        except Exception:
            cur_pos = (0.0, 0.0, 0.0)

        return {
            'instr_id': ep.episode_id,
            'scan': scan,
            'viewpoint': vp,
            'viewIndex': 0,  # not used by agent (only its own state)
            'heading': base_heading,
            'elevation': base_elevation,
            'position': cur_pos,
            'feature': feature,
            'candidate': candidate,
            'obj_img_fts': obj_img_fts,
            'obj_ang_fts': obj_ang_fts,
            'obj_box_fts': obj_box_fts,
            'obj_ids': obj_ids,
            'instruction': ep.instruction.instruction_text,
            'instr_encoding': list(ep.instruction.instruction_tokens or []),
            'gt_path': list(ep.reference_viewpoints),
            'gt_end_vps': list(ep.target_visible_viewpoints or []),
            'gt_obj_id': ep.target_object_id,
            'distance': 0.0,  # filled in lazily by agent if needed
        }

    def _build_candidates(
        self, scan: str, vp: str, base_heading: float, base_elevation: float
    ) -> List[Dict[str, Any]]:
        """For each neighbor viewpoint of ``vp``, build the candidate dict
        using pre-computed rel angles + the corresponding view's ViT feature.
        """
        neighbors = self.dataset.get_candidates(scan, vp)
        view_fts = self.dataset.get_view_features(scan, vp)  # (36, vit)
        candidates: List[Dict[str, Any]] = []
        for nbr in neighbors:
            rel = self.dataset.get_rel_angle(scan, vp, nbr)
            if rel is None:
                # Fallback: place at view 0 with zero angles
                view_idx, dist, rel_h, rel_e = 0, 0.0, 0.0, 0.0
            else:
                view_idx, dist, rel_h, rel_e = rel
            # Adjust by base heading/elevation of current agent state
            abs_h = rel_h + base_heading
            abs_e = rel_e + base_elevation
            ang = angle_feature(abs_h, abs_e, self.angle_feat_size)
            cand_feat = np.concatenate(
                [view_fts[view_idx], ang], axis=0
            )  # (vit+ang,)
            try:
                pos = self.dataset.get_viewpoint_position(scan, nbr)
            except Exception:
                pos = (0.0, 0.0, 0.0)
            candidates.append({
                'scanId': scan,
                'viewpointId': nbr,
                'pointId': int(view_idx),
                'heading': abs_h,
                'elevation': abs_e,
                'distance': float(dist),
                'feature': cand_feat,
                'position': tuple(pos),
            })
        return candidates

    def _build_object_features(
        self, scan: str, vp: str, base_heading: float, base_elevation: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        obj = self.dataset.get_object_features(scan, vp)
        fts = obj.get('fts', np.zeros((0, self.obj_feat_size), dtype=np.float32))
        centers = obj.get('centers', [])
        bboxes = obj.get('bboxes', [])
        obj_ids = obj.get('obj_ids', [])

        n = len(fts) if hasattr(fts, '__len__') else 0
        if self.max_objects is not None and n > self.max_objects:
            fts = fts[: self.max_objects]
            centers = centers[: self.max_objects]
            bboxes = bboxes[: self.max_objects]
            obj_ids = obj_ids[: self.max_objects]
            n = self.max_objects

        obj_ang = np.zeros((n, self.angle_feat_size), dtype=np.float32)
        obj_box = np.zeros((n, 3), dtype=np.float32)
        for k in range(n):
            if k < len(centers):
                h, e = centers[k][0], centers[k][1]
                obj_ang[k] = angle_feature(
                    h - base_heading, e - base_elevation, self.angle_feat_size
                )
            if k < len(bboxes) and len(bboxes[k]) >= 4:
                _, _, w, h = bboxes[k]
                # Normalize by pano image dims (224 x 224)
                obj_box[k, 0] = h / 224.0
                obj_box[k, 1] = w / 224.0
                obj_box[k, 2] = obj_box[k, 0] * obj_box[k, 1]
        return fts, obj_ang, obj_box, list(obj_ids)

    # ----- nav graph distances -----

    @property
    def shortest_distances(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Nested dict ``{scan: {vp1: {vp2: dist}}}``. Lazy per-scan."""
        return self._ShortestDistancesView(self)

    class _ShortestDistancesView:
        """Proxy that computes all-pairs distances for a scan on first access."""

        def __init__(self, parent: "EpisodeBatch"):
            self._parent = parent

        def __getitem__(self, scan: str) -> Dict[str, Dict[str, float]]:
            if scan not in self._parent._shortest_distances_cache:
                G = self._parent.dataset._build_nav_graph(scan)
                # NetworkX all_pairs_dijkstra_path_length — weight from edge 'weight'
                # attribute if present, else hop count. Our nav graph has no weight
                # set, so we use euclidean distance between positions as weight.
                for u, v in G.edges():
                    pu = G.nodes[u]['position']
                    pv = G.nodes[v]['position']
                    G[u][v]['weight'] = float(np.linalg.norm(
                        np.array(pu) - np.array(pv)
                    ))
                lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
                # Flatten into nested dict-of-dicts
                flat = {
                    u: {v: d for v, d in dists.items()}
                    for u, dists in lengths.items()
                }
                self._parent._shortest_distances_cache[scan] = flat
            return self._parent._shortest_distances_cache[scan]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@baseline_registry.register_trainer(name="duet_il")
class DUETTrainer(BaseTrainer):
    """Behavior-cloning trainer for DUET on HM3D-AutoVLN.

    Inherits :class:`BaseTrainer` to satisfy the registry's subclass check,
    but does NOT use :class:`RolloutStorage` or :class:`PPOTrainer` — DUET's
    variable-candidate-count per step is incompatible with those fixed-shape
    abstractions. The train/eval methods below are self-contained.
    """

    supported_tasks: List[str] = ["VLNGraphNav-v0"]

    def __init__(self, config: "Any") -> None:
        self.config = config
        self.device = torch.device(
            f"cuda:{config.habitat_baselines.torch_gpu_id}"
            if torch.cuda.is_available() else "cpu"
        )

    # ----- entry points required by BaseTrainer -----

    def train(self) -> None:
        from navdsl.data_adapter.hm3d_autovln_dataset import HM3DAutoVLNDatasetV1
        from navdsl.policy.duet.agent_obj import GMapObjectNavAgent

        # 1. Build dataset (provides LMDB + nav graph access)
        ds_config = self.config.habitat.dataset
        train_dataset = HM3DAutoVLNDatasetV1(ds_config)
        print(f"[duet_il] train episodes: {len(train_dataset.episodes)}")

        # 2. Build args namespace for agent
        model_cfg = self.config.habitat_baselines.il.duet
        args = _ArgsNamespace(model_cfg)

        # 3. Build env adapter (provides obs format the agent expects)
        env = EpisodeBatch(
            episodes=train_dataset.episodes,
            dataset=train_dataset,
            batch_size=args.batch_size,
            angle_feat_size=args.angle_feat_size,
            image_feat_size=args.image_feat_size,
            obj_feat_size=args.obj_feat_size,
            max_objects=getattr(args, 'max_objects', None),
            seed=args.seed,
        )

        # 4. Build agent — internally constructs VLNBert + critic + optimizer
        self.agent = GMapObjectNavAgent(args, env, rank=0)

        # 5. Training loop — agent.train(n_iters) handles rollout + backward + step
        ckpt_dir = self.config.habitat_baselines.checkpoint_folder
        os.makedirs(ckpt_dir, exist_ok=True)
        steps_per_epoch = max(1, len(train_dataset.episodes) // args.batch_size)

        print(f"[duet_il] {steps_per_epoch} batches/epoch, "
              f"{args.max_epochs} epochs, batch_size={args.batch_size}, "
              f"train_alg={args.train_alg}")

        for epoch in range(args.max_epochs):
            # agent.train() iterates n_iters internally; each iter = one rollout
            self.agent.train(steps_per_epoch, feedback='teacher')

            il_losses = self.agent.logs.get('IL_loss', [])
            og_losses = self.agent.logs.get('OG_loss', [])
            avg_il = sum(il_losses) / max(1, len(il_losses))
            avg_og = sum(og_losses) / max(1, len(og_losses))
            print(f"[duet_il] epoch {epoch+1}/{args.max_epochs} done | "
                  f"IL_loss={avg_il:.4f} OG_loss={avg_og:.4f}")

            if (epoch + 1) % model_cfg.eval_interval == 0:
                self._save_checkpoint(ckpt_dir, epoch + 1)

        print("[duet_il] training complete")

    def eval(self) -> None:
        from habitat_baselines.utils.common import poll_checkpoint_folder
        ckpt_dir = self.config.habitat_baselines.eval_ckpt_path_dir
        for ckpt_path in poll_checkpoint_folder(ckpt_dir, "eval"):
            self._eval_checkpoint(ckpt_path)

    def _eval_checkpoint(self, checkpoint_path, *args, **kwargs):
        print(f"[duet_il] eval {checkpoint_path} — TODO")
        # TODO: load checkpoint, run inference loop, compute SPL/Success

    def _evaluate(self, dataset) -> None:
        print("[duet_il] inline _evaluate TODO")

    def _save_checkpoint(self, ckpt_dir: str, epoch: int) -> None:
        path = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
        torch.save(
            {
                "model_state_dict": self.agent.vln_bert.state_dict(),
                "optimizer_state_dict": self.agent.vln_bert_optimizer.state_dict(),
                "epoch": epoch,
            },
            path,
        )
        print(f"[duet_il] saved {path}")

    def save_checkpoint(self, name, *args, **kwargs):
        ckpt_dir = self.config.habitat_baselines.checkpoint_folder
        os.makedirs(ckpt_dir, exist_ok=True)
        self._save_checkpoint(ckpt_dir, 0)

    def load_checkpoint(self, ckpt_path, *args, **kwargs):
        state = torch.load(ckpt_path, map_location=self.device)
        sd = state.get('model_state_dict', state)
        if hasattr(self, 'agent'):
            self.agent.vln_bert.load_state_dict(sd)
        else:
            print("[duet_il] WARN: load_checkpoint called before train(); stashing state")
            self._pending_state = sd


class _ArgsNamespace:
    """Wrap a DictConfig so DUET code can read attributes (args.X).

    Adds defaults for fields that the original argparse namespace always
    provided but the NavDSL yaml may not specify.
    """

    _DEFAULTS = dict(
        # DUET model arch defaults
        tokenizer='bert',
        num_l_layers=9, num_pano_layers=2, num_x_layers=4,
        graph_sprels=True, fusion='dynamic',
        fix_lang_embedding=False, fix_pano_embedding=False, fix_local_branch=False,
        feat_dropout=0.0, dropout=0.1,
        # Training defaults
        optim='adamW', lr=1e-5, world_size=1,
        ignoreid=-1, max_action_len=20,
        enc_full_graph=True, loss_nav_3=True,
        detailed_output=False,
        expl_max_ratio=0.0,
        bert_ckpt_file=None,
        # Image/obj dims
        image_feat_size=768, obj_feat_size=768, angle_feat_size=4,
        # agent.train() needs these:
        train_alg='imitation',  # 'imitation' | 'dagger' | 'rehps'
        ml_weight=0.0,           # only used when train_alg != 'imitation'
        dagger_sample='sample',  # DAgger feedback mode
        aug=None,                # path to augmented data or None
    )

    def __init__(self, model_cfg):
        # Apply defaults first, then override with model_cfg values
        for k, v in self._DEFAULTS.items():
            setattr(self, k, v)
        if model_cfg is not None:
            for k in dir(model_cfg):
                if k.startswith('_'):
                    continue
                setattr(self, k, getattr(model_cfg, k))
        # Common alias: lr → learning_rate (yaml uses learning_rate)
        if not hasattr(self, 'lr') or self.lr is None:
            self.lr = getattr(model_cfg, 'learning_rate', 1e-5)
