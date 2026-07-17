#!/usr/bin/env python3
"""Sensor: candidate (adjacent) viewpoints with relative angles.

Output is a dict:
  {
    "vp_ids": List[str]                              # candidate viewpoint ids
    "rel_angles": ndarray (N, 4)                     # (view_idx, dist, rel_h, rel_e)
    "positions":   ndarray (N, 3)                    # world (X, Y, Z) of each candidate
    "mask":        ndarray (N,) bool                 # always True here; padding handled by policy
  }
"""
from typing import Any, Dict, List

import numpy as np
from gym import spaces

from habitat.core.registry import registry
from habitat.core.simulator import Observations, Sensor


@registry.register_sensor(name="CandidateViewpointsSensor")
class CandidateViewpointsSensor(Sensor):
    """Exposes the list of navigable neighbours of the current viewpoint,
    along with their relative angles (precomputed in
    scanvp_candview_relangles.json) and absolute positions.
    """

    def __init__(self, sim, config, dataset, task, name, **kwargs: Any):
        self._sim = sim
        self._dataset = dataset
        self._task = task
        self.uuid = config.get("uuid", "candidate_viewpoints")
        self.observation_space = spaces.Dict({
            "vp_ids": spaces.Discrete(n=1),  # placeholder; actual return is a list
            "rel_angles": spaces.Box(low=-np.inf, high=np.inf, shape=(0, 4), dtype=np.float32),
            "positions": spaces.Box(low=-np.inf, high=np.inf, shape=(0, 3), dtype=np.float32),
            "mask": spaces.Box(low=0, high=1, shape=(0,), dtype=bool),
        })

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_observation_space(self) -> spaces.Dict:
        return self.observation_space

    def get_observation(
        self,
        observations: Observations,
        episode,
        task,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        scan = episode.scene_scan_id
        cur_vp = task.current_viewpoint_id
        candidates: List[str] = self._dataset.get_candidates(scan, cur_vp)

        rel_angles = np.zeros((len(candidates), 4), dtype=np.float32)
        positions = np.zeros((len(candidates), 3), dtype=np.float32)
        for i, vp in enumerate(candidates):
            rel = self._dataset.get_rel_angle(scan, cur_vp, vp)
            if rel is not None:
                rel_angles[i] = np.asarray(rel, dtype=np.float32)
            positions[i] = self._dataset.get_viewpoint_position(scan, vp)

        return {
            "vp_ids": list(candidates),
            "rel_angles": rel_angles,
            "positions": positions,
            "mask": np.ones(len(candidates), dtype=bool),
        }
