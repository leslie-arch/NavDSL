#!/usr/bin/env python3
"""Sensor: 36-view ViT features for the current viewpoint."""
from typing import Any

import numpy as np
from gym import spaces

from habitat.core.registry import registry
from habitat.core.simulator import Observations, Sensor


@registry.register_sensor(name="ViewpointFeatureSensor")
class ViewpointFeatureSensor(Sensor):
    """Reads (36, 768) ViT-B/16 panorama features for the current viewpoint
    from the view LMDB. The 36 views are 12 heading x 3 elevation.
    """

    def __init__(self, sim, config, dataset, task, name, **kwargs: Any):
        self._sim = sim
        self._dataset = dataset
        self._task = task
        self.uuid = config.get("uuid", "viewpoint_features")
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(36, 768), dtype=np.float32
        )

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_observation_space(self) -> spaces.Box:
        return self.observation_space

    def get_observation(
        self,
        observations: Observations,
        episode,
        task,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        scan = episode.scene_scan_id
        vp = task.current_viewpoint_id
        return self._dataset.get_view_features(scan, vp)
