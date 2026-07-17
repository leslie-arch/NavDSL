#!/usr/bin/env python3
"""Sensor: object features (and metadata) for the current viewpoint.

Output is a dict with keys: fts (N, 768), obj_ids, view_ids, obj_names,
bboxes, centers, 3d_centers, 3d_sizes. N is variable per viewpoint.
"""
from typing import Any, Dict

import numpy as np
from gym import spaces

from habitat.core.registry import registry
from habitat.core.simulator import Observations, Sensor


@registry.register_sensor(name="ObjectFeatureSensor")
class ObjectFeatureSensor(Sensor):
    """Reads object features for the current viewpoint from the object LMDB."""

    def __init__(self, sim, config, dataset, task, name, **kwargs: Any):
        self._sim = sim
        self._dataset = dataset
        self._task = task
        self.uuid = config.get("uuid", "object_features")
        # Use a permissive Dict space; N is variable.
        self.observation_space = spaces.Dict({
            "fts": spaces.Box(low=-np.inf, high=np.inf, shape=(0, 768), dtype=np.float32),
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
        vp = task.current_viewpoint_id
        return self._dataset.get_object_features(scan, vp)
