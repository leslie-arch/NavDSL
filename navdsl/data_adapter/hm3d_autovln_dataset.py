#!/usr/bin/env python3
"""Dataset class for HM3D-AutoVLN graph-nav VLN task.

Registered as ``"HM3DAutoVLN-v1"``. Does NOT extend VLNDatasetV1 (avoids the
forced instruction_vocab requirement); extends habitat's base Dataset directly.

Provides:
  * Episode loading from {split}.json.gz produced by convert_hm3d_autovln.py
  * Per-scene nav graph lookup (networkx.Graph built from {scan}_connectivity.json)
  * View feature LMDB reader: get_view_features(scan, vp) -> (36, 768)
  * Object feature LMDB reader: get_object_features(scan, vp) -> dict
  * Candidate angle lookup from scanvp_candview_relangles.json
"""
import gzip
import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import attr
import networkx as nx
import numpy as np

from habitat.core.dataset import ALL_SCENES_MASK, Dataset
from habitat.core.registry import registry
from habitat.core.utils import not_none_validator
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
)
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.vln.vln import InstructionData, VLNEpisode

if TYPE_CHECKING:
    from omegaconf import DictConfig

# Default scene prefix; stripped and re-joined with scenes_dir.
DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"

# Slicing: LMDB stores 1768-dim features; first 768 = ViT-B/16 (per
# view_ft_size in HM3DAutoVLN/reverie_speaker_src/gen_hm3d_captions_bbox2d.py:33).
VIEW_VIT_DIM = 768
OBJ_VIT_DIM = 768


@attr.s(auto_attribs=True, kw_only=True)
class HM3DAutoVLNEpisode(VLNEpisode):
    """VLNEpisode + AutoVLN-specific viewpoint metadata."""

    scene_scan_id: str = attr.ib(default=None, validator=not_none_validator)
    start_viewpoint_id: str = attr.ib(
        default=None, validator=not_none_validator
    )
    reference_viewpoints: List[str] = attr.ib(
        default=None, validator=not_none_validator
    )
    goal_viewpoint_id: str = attr.ib(default=None, validator=not_none_validator)
    target_object_id: Optional[str] = attr.ib(default=None)
    target_visible_viewpoints: List[str] = attr.ib(default=None)


def _open_lmdb(path: str, subdir: bool = True):
    """Open LMDB read-only. Imported lazily so the dataset module can be loaded
    in environments without lmdb (e.g. when only doing config introspection).

    Args:
        path: LMDB file or directory path.
        subdir: True for directory layout (path/data.mdb), False for single
               file layout (path IS the data file).
    """
    import lmdb

    return lmdb.open(
        path, readonly=True, lock=False, readahead=False, meminit=False,
        subdir=subdir,
    )


def _unpack(raw: bytes):
    import msgpack
    import msgpack_numpy

    msgpack_numpy.patch()
    return msgpack.unpackb(raw)


@registry.register_dataset(name="HM3DAutoVLN-v1")
class HM3DAutoVLNDatasetV1(Dataset):
    r"""Loads HM3D-AutoVLN-generated VLN episodes with full nav graph +
    feature access. Episodes are produced by
    ``navdsl.data_adapter.convert_hm3d_autovln``.
    """

    episodes: List[HM3DAutoVLNEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    # ------- construction -------

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        self.episodes = []
        self._nav_graph_cache: Dict[str, nx.Graph] = {}
        self._rel_angles: Dict[str, Dict[str, Tuple[int, float, float, float]]] = {}
        # Legacy: single-large-LMDB mode
        self._view_env = None
        self._obj_env = None
        # New: per-scan-LMDB mode
        self._per_scan_view_dir = ""
        self._per_scan_obj_dir = ""
        self._per_scan_envs: Dict[Tuple[str, str], Any] = {}  # (kind, scan) -> env
        self._per_scan_cache_max = 64  # max cached envs (LRU eviction)
        self._nav_graph_dir_path = ""

        if config is None:
            return

        # 0. Resolve nav_graph_dir (used by _build_nav_graph)
        nav_graph_dir = getattr(config, "nav_graph_dir", None)
        if nav_graph_dir:
            self._nav_graph_dir_path = nav_graph_dir

        # 1. Load rel_angles (one big json, ~MB-scale; load once)
        rel_angles_path = getattr(config, "rel_angles_path", None)
        if rel_angles_path and os.path.isfile(rel_angles_path):
            with open(rel_angles_path) as f:
                self._rel_angles = json.load(f)

        # 2. Feature storage: support two modes.
        #    Legacy (single large LMDB):
        #      view_feature_lmdb: /path/to/view_timm_imagenet_vitb16
        #      object_feature_lmdb: /path/to/obj2d_ade20k_pseudo3d_merged_timm_imagenet_vitb16
        #    New (per-scan LMDB directory):
        #      view_feature_lmdb_dir: /path/to/per_scan
        #      object_feature_lmdb_dir: /path/to/per_scan
        #    Per-scan mode takes precedence if both are set.
        per_scan_dir = getattr(config, "feature_per_scan_dir", None)
        if per_scan_dir and os.path.isdir(per_scan_dir):
            # Combined per-scan LMDB (recommended layout):
            #   {feature_per_scan_dir}/{split}/{scan}.lmdb
            # keys inside: 'view_{vp}' and 'obj_{vp}'
            self._per_scan_view_dir = per_scan_dir
            self._per_scan_obj_dir = per_scan_dir
        else:
            view_dir = getattr(config, "view_feature_lmdb_dir", None)
            obj_dir = getattr(config, "object_feature_lmdb_dir", None)
            if view_dir and os.path.isdir(view_dir):
                self._per_scan_view_dir = view_dir
            if obj_dir and os.path.isdir(obj_dir):
                self._per_scan_obj_dir = view_dir if view_dir else obj_dir

        # Fallback to legacy single-LMDB mode if per-scan not configured
        if not self._per_scan_view_dir:
            view_lmdb = getattr(config, "view_feature_lmdb", None)
            if view_lmdb and os.path.isdir(view_lmdb):
                self._view_env = _open_lmdb(view_lmdb)
        if not self._per_scan_obj_dir:
            obj_lmdb = getattr(config, "object_feature_lmdb", None)
            if obj_lmdb and os.path.isdir(obj_lmdb):
                self._obj_env = _open_lmdb(obj_lmdb)

        # 3. Load episodes json.gz
        dataset_path = config.data_path.format(split=config.split)
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(
                f"Episode file not found: {dataset_path}. "
                f"Run navdsl.data_adapter.convert_hm3d_autovln first."
            )
        with gzip.open(dataset_path, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.scenes_dir)

        # 4. Filter by content_scenes
        scenes_filter = self.build_content_scenes_filter(config)
        self.episodes = list(filter(scenes_filter, self.episodes))

    # ------- habitat API -------

    @staticmethod
    def check_config_paths_exist(config: "DictConfig") -> bool:
        return os.path.exists(config.data_path.format(split=config.split)) and (
            os.path.isdir(config.scenes_dir) if hasattr(config, "scenes_dir") else True
        )

    @classmethod
    def build_content_scenes_filter(cls, config: "DictConfig"):
        """Match by AutoVLN ``scene_scan_id`` (e.g. ``00000-kfPV7w3FaU5``),
        not by the basename of ``scene_id``. The upstream filter would return
        ``<scan>.basis`` from ``.../<scan>/<scan>.basis.glb`` and miss the
        plain-scan content_scenes entries callers typically pass.
        """
        scenes_to_load = set(config.content_scenes)

        def _filter(ep: HM3DAutoVLNEpisode) -> bool:
            return (
                ALL_SCENES_MASK in scenes_to_load
                or getattr(ep, "scene_scan_id", None) in scenes_to_load
            )

        return _filter

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        for ep_dict in deserialized["episodes"]:
            ep = HM3DAutoVLNEpisode(**ep_dict)
            ep.episode_id = str(ep.episode_id)

            if scenes_dir is not None:
                if ep.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    ep.scene_id = ep.scene_id[len(DEFAULT_SCENE_PATH_PREFIX) :]
                ep.scene_id = os.path.join(scenes_dir, ep.scene_id)

            # Rehydrate nested dataclasses
            ep.instruction = InstructionData(**ep_dict["instruction"])
            ep.goals = [NavigationGoal(**g) for g in ep_dict.get("goals", [])]

            self.episodes.append(ep)

    # ------- nav graph -------

    def _build_nav_graph(self, scan: str) -> nx.Graph:
        """Load {scan}_connectivity.json, build an undirected graph with
        viewpoint positions on nodes. Cached per scan."""
        if scan in self._nav_graph_cache:
            return self._nav_graph_cache[scan]
        if not self._nav_graph_dir_path:
            raise RuntimeError(
                "nav_graph_dir not configured. Set it via the dataset config "
                "(habitat.dataset.nav_graph_dir)."
            )
        conn_path = os.path.join(
            self._nav_graph_dir_path, f"{scan}_connectivity.json"
        )
        with open(conn_path) as f:
            data = json.load(f)

        G = nx.Graph()
        vp_to_entry: Dict[str, Dict[str, Any]] = {}
        for entry in data:
            if not entry["included"]:
                continue
            vp_id = entry["image_id"]
            pose = entry["pose"]
            # Per convert_hm3d_autovln.parse_pose convention
            x, y, z = float(pose[3]), float(pose[11]), float(-pose[7])
            G.add_node(vp_id, position=(x, y, z))
            vp_to_entry[vp_id] = entry

        for vp_id, entry in vp_to_entry.items():
            for j, conn in enumerate(entry["unobstructed"]):
                if conn:
                    nbr_id = data[j]["image_id"]
                    if nbr_id in vp_to_entry:
                        G.add_edge(vp_id, nbr_id)

        self._nav_graph_cache[scan] = G
        return G

    def set_nav_graph_dir(self, path: str) -> None:
        """Public setter for the nav graph dir, in case the dataset was
        constructed without a config (e.g. tests)."""
        self._nav_graph_dir_path = path

    def get_viewpoint_position(
        self, scan: str, viewpoint_id: str
    ) -> Tuple[float, float, float]:
        G = self._build_nav_graph(scan)
        return G.nodes[viewpoint_id]["position"]

    def get_candidates(self, scan: str, viewpoint_id: str) -> List[str]:
        """Adjacent (navigable) viewpoint ids from a given viewpoint."""
        G = self._build_nav_graph(scan)
        return list(G.neighbors(viewpoint_id))

    def get_rel_angle(
        self, scan: str, viewpoint_id: str, neighbor_vp: str
    ) -> Optional[Tuple[int, float, float, float]]:
        """Returns (best_view_idx, distance, rel_heading, rel_elevation) or None."""
        key = f"{scan}_{viewpoint_id}"
        nbrs = self._rel_angles.get(key, {})
        entry = nbrs.get(neighbor_vp)
        if entry is None:
            return None
        # entry is [view_idx, dist, rel_heading, rel_elevation]
        return tuple(entry)

    # ------- LMDB readers -------

    @staticmethod
    def _split_of_scan(scan: str) -> str:
        """Numeric prefix < 800 -> train, else val. Matches converter convention."""
        try:
            prefix = int(scan.split("-")[0])
        except (ValueError, IndexError):
            return "val"
        return "train" if prefix < 800 else "val"

    def _get_per_scan_env(self, kind: str, scan: str):
        """Lazily open a per-scan LMDB and cache it.

        In the recommended combined layout (one LMDB per scan, containing both
        view and obj keys), the env is shared between view and obj reads.
        The cache key is therefore just ``scan``, not ``(kind, scan)``.

        Args:
            kind: 'view' or 'obj' (selects which top-level dir to look in
                  when view/obj live in separate trees).
            scan: scene scan id (e.g., '00000-kfPV7w3FaU5').
        """
        cache_key = scan  # combined layout: same env for both kinds
        if cache_key in self._per_scan_envs:
            # LRU: move to end (insertion order)
            env = self._per_scan_envs.pop(cache_key)
            self._per_scan_envs[cache_key] = env
            return env

        # Open new env
        base_dir = self._per_scan_view_dir if kind == "view" else self._per_scan_obj_dir
        if not base_dir:
            raise RuntimeError(
                f"per-scan LMDB dir not configured (kind={kind}). "
                f"Set dataset.feature_per_scan_dir or view_feature_lmdb_dir/object_feature_lmdb_dir."
            )
        split = self._split_of_scan(scan)
        path = os.path.join(base_dir, split, f"{scan}.lmdb")
        if not os.path.isfile(path):
            # Some scans might not have obj features (e.g., obj2d empty for that scene).
            # Return None to let the caller handle the missing case.
            return None
        env = _open_lmdb(path, subdir=False)  # per-scan uses single-file layout
        self._per_scan_envs[cache_key] = env

        # LRU eviction
        while len(self._per_scan_envs) > self._per_scan_cache_max:
            old_key, old_env = next(iter(self._per_scan_envs.items()))
            old_env.close()
            del self._per_scan_envs[old_key]
        return env

    def _read_feature(self, kind: str, scan: str, viewpoint_id: str):
        """Unified reader. Returns raw msgpack-unpacked value, or None if missing."""
        # Per-scan mode
        if self._per_scan_view_dir or self._per_scan_obj_dir:
            env = self._get_per_scan_env(kind, scan)
            if env is None:
                return None
            key = f"{kind}_{viewpoint_id}".encode("ascii")
            with env.begin() as txn:
                raw = txn.get(key)
            if raw is None:
                return None
            return _unpack(raw)
        # Legacy single-LMDB mode
        if kind == "view":
            if self._view_env is None:
                raise RuntimeError("view_feature_lmdb not configured")
            key = f"{scan}_{viewpoint_id}".encode("ascii")
            with self._view_env.begin() as txn:
                raw = txn.get(key)
        else:  # obj
            if self._obj_env is None:
                raise RuntimeError("object_feature_lmdb not configured")
            key = f"{scan}_{viewpoint_id}".encode("ascii")
            with self._obj_env.begin() as txn:
                raw = txn.get(key)
        if raw is None:
            return None
        return _unpack(raw)

    def get_view_features(self, scan: str, viewpoint_id: str) -> np.ndarray:
        """Returns (36, VIEW_VIT_DIM) ViT features for the panorama at this viewpoint."""
        ft = self._read_feature("view", scan, viewpoint_id)
        if ft is None:
            return np.zeros((36, VIEW_VIT_DIM), dtype=np.float32)
        return ft[:, :VIEW_VIT_DIM].astype(np.float32)

    def get_object_features(self, scan: str, viewpoint_id: str) -> Dict[str, Any]:
        """Returns a dict with keys: obj_ids, view_ids, obj_names, bboxes,
        centers, fts (N, OBJ_VIT_DIM), 3d_centers, 3d_sizes. Empty if missing.
        """
        obj = self._read_feature("obj", scan, viewpoint_id)
        if obj is None:
            return {
                "obj_ids": [], "view_ids": [], "obj_names": [],
                "bboxes": [], "centers": [],
                "fts": np.zeros((0, OBJ_VIT_DIM), dtype=np.float32),
                "3d_centers": [], "3d_sizes": [],
            }
        # Slice fts to OBJ_VIT_DIM
        if "fts" in obj and hasattr(obj["fts"], "shape"):
            obj["fts"] = obj["fts"][:, :OBJ_VIT_DIM].astype(np.float32)
        return obj
