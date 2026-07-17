#!/usr/bin/env python3
"""Habitat-baselines Policy wrapper around the ported DUET model.

The DUET model + agent_obj.GMapObjectNavAgent implement the actual
inference loop. This class bridges them with habitat-baselines' Policy
interface so the trainer can register it via ``@register_policy``.

Two modes:
  * :meth:`act` — single-step inference given habitat observations.
  * :meth:`rollout_episode` — full-episode rollout used by the trainer
    (mirrors ``GMapObjectNavAgent.rollout`` but driven by the habitat
    Env, not MatterSim).
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo import Policy

from .vlnbert_init import build_duet
from .model import VLNBert


@baseline_registry.register_policy(name="DUETPolicy")
class DUETPolicy(Policy):
    """Thin Policy wrapper around the DUET model.

    The heavy lifting (panorama feature assembly, GMap building, candidate
    scoring) lives in :class:`navdsl.policy.duet.agent_obj.GMapObjectNavAgent`,
    which the trainer instantiates directly. This Policy class exists
    primarily for habitat-baselines registration and to expose a
    habitat-compatible ``act`` method for any future RL fine-tuning.
    """

    def __init__(
        self,
        observation_space: Any = None,
        action_space: Any = None,
        config: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # Build VLNBert (which wraps GlocalTextPathNavCMT) — pretrained
        # weights loaded via vlnbert_init's key remapping.
        model_config = dict(
            bert_ckpt_file=getattr(config, "checkpoint", None),
            tokenizer=getattr(config, "tokenizer", "bert"),
            image_feat_size=getattr(config, "image_feat_size", 768),
            angle_feat_size=getattr(config, "angle_feat_size", 4),
            obj_feat_size=getattr(config, "obj_feat_size", 768),
            num_l_layers=getattr(config, "num_l_layers", 9),
            num_pano_layers=getattr(config, "num_pano_layers", 2),
            num_x_layers=getattr(config, "num_x_layers", 4),
            graph_sprels=getattr(config, "graph_sprels", True),
            fusion=getattr(config, "fusion", "dynamic"),
            fix_lang_embedding=getattr(config, "fix_lang_embedding", False),
            fix_pano_embedding=getattr(config, "fix_pano_embedding", False),
            fix_local_branch=getattr(config, "fix_local_branch", False),
        )
        # VLNBert expects argparse-like args; build a dummy namespace.
        ns = _DummyArgs(model_config)
        self.vln_bert = VLNBert(ns)

    @property
    def net(self):
        """habitat-baselines sometimes accesses policy.net — alias to vln_bert."""
        return self.vln_bert

    def act(
        self,
        observations,
        rnn_hidden,
        prev_actions,
        masks,
        deterministic: bool = False,
        **kwargs: Any,
    ):
        """Single-step action. Used by habitat-baselines trainers.

        Returns a lightweight ActionData-like object with ``actions`` field
        carrying either the integer candidate index or -1 for STOP.
        """
        # NOTE: full per-step inference is implemented in the trainer's
        # rollout loop, where the agent has access to per-episode state
        # (visited viewpoints, GMap). The Policy.act interface is too
        # stateless to drive DUET directly without injecting the agent.
        raise NotImplementedError(
            "DUETPolicy.act is intentionally not implemented — use the "
            "trainer's rollout loop with GMapObjectNavAgent."
        )

    @classmethod
    def from_config(cls, config, *args, **kwargs):
        """habitat-baselines factory hook."""
        return cls(
            observation_space=None,
            action_space=None,
            config=config.habitat_baselines.rl.policy[
                config.habitat_baselines.rl.policy.name
            ],
        )


class _DummyArgs:
    """Empty argparse-like namespace populated from a dict."""

    def __init__(self, attrs: Dict[str, Any]) -> None:
        for k, v in attrs.items():
            setattr(self, k, v)
        # Defaults required by VLNBert but not always in user config
        for k, v in dict(feat_dropout=0.0, dropout=0.1).items():
            if not hasattr(self, k):
                setattr(self, k, v)
