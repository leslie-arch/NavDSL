#!/usr/bin/env python3
"""VLNGraphNav-v0: discrete-viewpoint graph navigation task for HM3D-AutoVLN.

Agent state is a viewpoint id (string). At each step the agent picks one of
the candidate viewpoints (adjacent to the current) or stops. GotoViewpoint
teleports the agent in habitat-sim (keeping habitat-sim in the loop for
navmesh validation, optional RGB rendering, and metric computation), while
the policy input is LMDB-cached ViT features keyed by (scan, viewpoint).

Task state (per-episode, mutable):
  * current_viewpoint_id : str
  * visited_viewpoints   : List[str]
  * current_episode      : HM3DAutoVLNEpisode
  * is_stop_called       : bool
  * step_count           : int
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gym import spaces

from habitat.core.embodied_task import EmbodiedTask, Episode
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import NavigationTask
from habitat.tasks.nav.nav import SimulatorTaskAction


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@registry.register_task_action
class GotoViewpointAction(SimulatorTaskAction):
    """Move agent to one of the candidate viewpoints (by index).

    Action argument: ``viewpoint_idx: int``.
    """

    name: str = "goto_viewpoint"

    def reset(self, task: EmbodiedTask, *args: Any, **kwargs: Any) -> None:
        # State init happens at task level; nothing to do here.
        pass

    def step(
        self,
        viewpoint_idx: int,
        task: EmbodiedTask,
        *args: Any,
        **kwargs: Any,
    ):
        ep = task.current_episode
        scan = ep.scene_scan_id
        cur_vp = task.current_viewpoint_id
        candidates: List[str] = task._dataset.get_candidates(scan, cur_vp)

        if not (0 <= viewpoint_idx < len(candidates)):
            # Out-of-range index — treat as no-op (lets the policy mask invalid
            # actions without crashing the env).
            return

        target_vp = candidates[viewpoint_idx]
        target_pos = task._dataset.get_viewpoint_position(scan, target_vp)

        # Keep habitat-sim in the loop: teleport the agent for real. Rotation is
        # irrelevant to DUET (panorama-based), use identity quaternion.
        try:
            self._sim.set_agent_state(
                target_pos, [0.0, 0.0, 0.0, 1.0], reset_sensors=False
            )
        except Exception:
            # If the navmesh rejects this position, swallow the error so the
            # episode continues with stale sim state. Position-tracked metrics
            # won't update but viewpoint-based metrics still work.
            pass

        task.current_viewpoint_id = target_vp
        task.visited_viewpoints.append(target_vp)
        task.step_count += 1

    @property
    def action_space(self) -> spaces.Dict:
        # action_args = {"viewpoint_idx": int}. Generous upper bound; actual
        # candidate count varies per step and is exposed by
        # CandidateViewpointsSensor. The policy must mask invalid indices.
        return spaces.Dict({"viewpoint_idx": spaces.Discrete(n=32)})


@registry.register_task_action
class StopAction(SimulatorTaskAction):
    """End the episode. Used by the policy when it believes the goal is reached."""

    name: str = "stop"

    def reset(self, task: EmbodiedTask, *args: Any, **kwargs: Any) -> None:
        task.is_stop_called = False  # type: ignore[attr-defined]

    def step(self, task: EmbodiedTask, *args: Any, **kwargs: Any) -> None:
        task.is_stop_called = True  # type: ignore[attr-defined]
        task.step_count += 1

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(n=1)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@registry.register_task(name="VLNGraphNav-v0")
class VLNGraphNavTask(NavigationTask):
    r"""Discrete-viewpoint VLN task backed by HM3D-AutoVLN nav graphs.

    State initialized on :meth:`reset` from the episode's ``start_viewpoint_id``.
    The task is considered active while neither stop is called nor the step
    budget exceeds ``max_episode_steps``.
    """

    def __init__(
        self,
        config: "Any",
        sim: Simulator,
        dataset: Optional["Any"] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset, **kwargs)

        # Mutable per-episode state
        self.current_viewpoint_id: str = ""
        self.visited_viewpoints: List[str] = []
        self.current_episode: Optional[Episode] = None
        self.is_stop_called: bool = False
        self.step_count: int = 0
        # Max steps read from habitat.environment.max_episode_steps (set in
        # benchmark yaml). Default 20 — typical for graph-nav VLN where avg
        # path length is ~5 viewpoints.
        # The config passed to __init__ is the task subtree; env max_steps is
        # at the parent level, so we read via attribute access on the parent
        # structure if available, else default.
        env_cfg = getattr(getattr(config, "_parent_", None), "environment", None) \
            if hasattr(config, "_parent_") else None
        self._max_steps: int = 20
        if env_cfg is not None and hasattr(env_cfg, "max_episode_steps"):
            self._max_steps = int(env_cfg.max_episode_steps)

    def reset(self, episode: Episode):
        # Initialize viewpoint state BEFORE sensor_suite.get_observations fires
        # (sensors read task.current_viewpoint_id).
        self.current_episode = episode
        self.current_viewpoint_id = episode.start_viewpoint_id
        self.visited_viewpoints = [episode.start_viewpoint_id]
        self.is_stop_called = False
        self.step_count = 0

        # Drive the parent reset, which calls sim.reset() (puts the agent at
        # agent_cfg.start_position, already set via overwrite_sim_config to
        # episode.start_position which equals the start viewpoint's pose) and
        # then sensor_suite.get_observations + actions.reset.
        return super().reset(episode)

    def overwrite_sim_config(self, config: Any, episode: Episode) -> Any:
        """Configure the sim to load this episode's scene + start state."""
        from habitat.config import read_write

        with read_write(config):
            config.simulator.scene = episode.scene_id
            agent_cfg = config.simulator.agents.main_agent
            agent_cfg.start_position = list(episode.start_position)
            agent_cfg.start_rotation = [float(k) for k in episode.start_rotation]
            agent_cfg.is_set_start_state = True
        return config

    def _check_episode_is_active(
        self, *args: Any, action: Dict[str, Any], episode: Episode, **kwargs: Any
    ) -> bool:
        if self.is_stop_called:
            return False
        if self.step_count >= self._max_steps:
            return False
        return True
