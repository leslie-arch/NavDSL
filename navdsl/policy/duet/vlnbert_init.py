#!/usr/bin/env python3
"""DUET model factory: build GlocalTextPathNavCMT and load pretrained weights.

Adapted from HM3DAutoVLN/map_nav_src/models/vlnbert_init.py — kept the
state_dict key remapping (strip 'module.' prefix, prepend 'bert.' for
head/sap_fuse keys) so the released `model_step_35000.pt` loads cleanly.

Key change vs. upstream: ``build_duet(...)`` accepts a plain dict (or
DictConfig) instead of argparse Namespace, so it integrates with habitat's
OmegaConf-based config flow.
"""
from typing import Any, Dict, Optional

import torch


# Defaults match HM3DAutoVLN/REVERIE/configs/reverie_obj_model_config.json
# (CMT-VLN-BERT, ViT-B/16 image encoder).
DEFAULTS: Dict[str, Any] = dict(
    bert_ckpt_file=None,           # set this to model_step_35000.pt path
    tokenizer="bert",              # 'bert' or 'xlm'
    image_feat_size=768,
    angle_feat_size=4,
    obj_feat_size=768,
    num_l_layers=9,
    num_pano_layers=2,
    num_x_layers=4,
    graph_sprels=True,
    fusion="dynamic",
    fix_lang_embedding=False,
    fix_pano_embedding=False,
    fix_local_branch=False,
)


def get_tokenizer(tokenizer_name: str = "bert"):
    """Returns a HuggingFace tokenizer (BERT or XLM-R)."""
    from transformers import AutoTokenizer
    if tokenizer_name == "xlm":
        cfg_name = "xlm-roberta-base"
    else:
        cfg_name = "bert-base-uncased"
    return AutoTokenizer.from_pretrained(cfg_name)


def build_duet(config: Optional[Dict[str, Any]] = None, **overrides: Any):
    """Construct the DUET visual-language navigator and (optionally) load
    pretrained weights.

    Args:
        config: dict-like with keys from DEFAULTS. None uses defaults.
        **overrides: keyword overrides applied on top of ``config``.

    Returns:
        GlocalTextPathNavCMT instance with weights loaded.
    """
    cfg = dict(DEFAULTS)
    if config is not None:
        cfg.update(dict(config))
    cfg.update(overrides)

    from transformers import PretrainedConfig
    from .vilmodel import GlocalTextPathNavCMT

    model_path = cfg["bert_ckpt_file"]
    new_ckpt_weights: Dict[str, torch.Tensor] = {}
    if model_path is not None:
        ckpt = torch.load(model_path, map_location="cpu")
        # The original training script uses DDP, so keys have 'module.' prefix.
        for k, v in ckpt.items():
            if k.startswith("module"):
                k = k[7:]
            if "_head" in k or "sap_fuse" in k:
                new_ckpt_weights["bert." + k] = v
            else:
                new_ckpt_weights[k] = v

    cfg_name = (
        "xlm-roberta-base" if cfg["tokenizer"] == "xlm" else "bert-base-uncased"
    )
    vis_config = PretrainedConfig.from_pretrained(cfg_name)
    if cfg["tokenizer"] == "xlm":
        vis_config.type_vocab_size = 2

    vis_config.max_action_steps = 100
    vis_config.image_feat_size = cfg["image_feat_size"]
    vis_config.angle_feat_size = cfg["angle_feat_size"]
    vis_config.obj_feat_size = cfg["obj_feat_size"]
    vis_config.obj_loc_size = 3
    vis_config.num_l_layers = cfg["num_l_layers"]
    vis_config.num_pano_layers = cfg["num_pano_layers"]
    vis_config.num_x_layers = cfg["num_x_layers"]
    vis_config.graph_sprels = cfg["graph_sprels"]
    vis_config.glocal_fuse = cfg["fusion"] == "dynamic"
    vis_config.fix_lang_embedding = cfg["fix_lang_embedding"]
    vis_config.fix_pano_embedding = cfg["fix_pano_embedding"]
    vis_config.fix_local_branch = cfg["fix_local_branch"]
    vis_config.update_lang_bert = not cfg["fix_lang_embedding"]
    vis_config.output_attentions = True
    vis_config.pred_head_dropout_prob = 0.1
    vis_config.use_lang2visn_attn = False

    return GlocalTextPathNavCMT.from_pretrained(
        pretrained_model_name_or_path=None,
        config=vis_config,
        state_dict=new_ckpt_weights,
    )


# Compatibility shim for upstream code that calls ``get_vlnbert_models(args)``.
def get_vlnbert_models(args, config=None):
    """Translate argparse-style ``args`` (with attributes) into a dict for
    :func:`build_duet`. Allows the ported ``models/model.py`` and
    ``reverie/agent_obj.py`` to keep their original call sites."""
    attrs = {}
    for k in DEFAULTS.keys():
        if hasattr(args, k):
            attrs[k] = getattr(args, k)
    return build_duet(attrs)
