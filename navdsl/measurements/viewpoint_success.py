#!/usr/bin/env python3
"""Measurements: viewpoint-based success and SPL for VLNGraphNav-v0.

ViewpointSuccess: 1.0 if the agent stops at a viewpoint where the target
object is visible (i.e. ``current_vp in episode.target_visible_viewpoints``),
0 otherwise.

ViewpointSPL: success * reference_path_viewpoint_count / max(reference,
actual_visited) — standard SPL formula but in viewpoint-count units.

ViewpointNE: number of steps (viewpoint transitions) taken, for logging.
"""
from typing import Any, Optional

from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.core.embodied_task import EmbodiedTask, Measure


@registry.register_measure(name="ViewpointSuccess")
class ViewpointSuccess(Measure):
    """1.0 if agent stops at a viewpoint where target object is visible."""

    cls_uuid: str = "viewpoint_success"

    def __init__(
        self,
        sim: Simulator,
        dataset,
        task: EmbodiedTask,
        config,
        *args: Any,
        **kwargs: Any,
    ):
        self._task = task
        super().__init__()

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any) -> str:
        return ViewpointSuccess.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._metric = 0.0

    def update_metric(
        self, episode, task, *args: Any, observations=None, **kwargs: Any
    ):
        # Only meaningful once the agent has stopped (or episode ended). The
        # metric is recomputed on every step so the latest value reflects the
        # final state when the episode ends.
        target_vps = set(getattr(episode, "target_visible_viewpoints", []) or [])
        if task.is_stop_called and task.current_viewpoint_id in target_vps:
            self._metric = 1.0
        else:
            # Carry the last computed value; we only flip to 1.0 on success
            # and reset to 0.0 on episode reset.
            pass


@registry.register_measure(name="ViewpointSPL")
class ViewpointSPL(Measure):
    """Success weighted by (viewpoint-count) path length.

    SPL = success * ref_len / max(ref_len, actual_len)
    """

    cls_uuid: str = "viewpoint_spl"

    def __init__(
        self,
        sim: Simulator,
        dataset,
        task: EmbodiedTask,
        config,
        *args: Any,
        **kwargs: Any,
    ):
        self._task = task
        self._success_measure_name = config.get("success_measure", "ViewpointSuccess")
        super().__init__()

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any) -> str:
        return ViewpointSPL.cls_uuid

    def reset_metric(self, episode, task, observations, *args: Any, **kwargs: Any):
        self._metric = 0.0
        # Wire up dependency on ViewpointSuccess so update order is correct.
        task.measurements[self._success_measure_name].reset_metric(
            episode, task, observations=observations
        )

    def update_metric(
        self, episode, task, observations, *args: Any, **kwargs: Any
    ):
        # Update the success measurement first.
        task.measurements[self._success_measure_name].update_metric(
            episode, task, observations=observations
        )
        success = task.measurements[self._success_measure_name].get_metric()

        ref_len = max(1, len(getattr(episode, "reference_viewpoints", [])) - 1)
        actual_len = max(1, len(task.visited_viewpoints) - 1)
        self._metric = success * ref_len / max(ref_len, actual_len)


@registry.register_measure(name="ViewpointSteps")
class ViewpointSteps(Measure):
    """Number of viewpoint transitions taken. For logging only."""

    cls_uuid: str = "viewpoint_steps"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any) -> str:
        return ViewpointSteps.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._metric = 0

    def update_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._metric = task.step_count
