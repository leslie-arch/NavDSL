#!/usr/bin/env python3
"""Sensor: global graph state for DUET's GMap branch.

Exposes the topology of explored nodes — visited viewpoints plus their
unvisited neighbours — as id lists. The DUET policy (Phase 4) is responsible
for assembling per-node features and running graph attention.

Output:
  {
    "visited_vp_ids":   List[str]    # all visited viewpoints in order
    "frontier_vp_ids":  List[str]    # unvisited neighbours of any visited vp
    "current_vp_id":    str          # current viewpoint id
    "edges":            List[Tuple[str, str]]   # adjacency within visited+frontier
  }
"""
from typing import Any, Dict, List, Tuple

from gym import spaces

from habitat.core.registry import registry
from habitat.core.simulator import Observations, Sensor


@registry.register_sensor(name="GraphNodesSensor")
class GraphNodesSensor(Sensor):
    """Exposes the explored subgraph (visited + frontier) for DUET's global
    graph attention branch."""

    def __init__(self, sim, config, dataset, task, name, **kwargs: Any):
        self._sim = sim
        self._dataset = dataset
        self._task = task
        self.uuid = config.get("uuid", "graph_nodes")
        self.observation_space = spaces.Dict({
            "visited_vp_ids": spaces.Discrete(n=1),
            "frontier_vp_ids": spaces.Discrete(n=1),
            "current_vp_id": spaces.Discrete(n=1),
            "edges": spaces.Discrete(n=1),
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
        visited: List[str] = list(task.visited_viewpoints)
        visited_set = set(visited)

        # Frontier = any unvisited neighbour of any visited vp
        frontier_set = set()
        edges: List[Tuple[str, str]] = []
        for v in visited:
            try:
                nbrs = self._dataset.get_candidates(scan, v)
            except Exception:
                nbrs = []
            for nbr in nbrs:
                edges.append((v, nbr))
                if nbr not in visited_set:
                    frontier_set.add(nbr)

        # Dedupe edges (graph is undirected)
        seen = set()
        unique_edges = []
        for a, b in edges:
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            unique_edges.append((a, b))

        return {
            "visited_vp_ids": visited,
            "frontier_vp_ids": sorted(frontier_set),
            "current_vp_id": task.current_viewpoint_id,
            "edges": unique_edges,
        }
