"""Structured-config schema for the HM3D-AutoVLN graph-nav VLN task.

Habitat's default `DatasetConfig` only exposes the basic PointNav-style fields
(`type`, `split`, `data_path`, `scenes_dir`, `content_scenes`, `metadata`).
The HM3D-AutoVLN dataset needs extra path fields for the nav-graph
connectivity, per-scan LMDB features, and the relative-angle lookup table.
Under Hydra's structured mode, any key that doesn't appear on the registered
schema is rejected with `KeyError`, so this module extends `DatasetConfig`
with the AutoVLN-specific fields and registers the new schema in the
ConfigStore under group `navdsl/dataset`, name `hm3d_autovln_schema`.

The yaml at `config/navdsl/dataset/vln/hm3d_autovln.yaml` references
`hm3d_autovln_schema` instead of the upstream `dataset_config_schema`; that
keeps every yaml key on the schema while staying inside Hydra's strict mode.
The schema lives under the `navdsl/` Hydra group (decoupled from the yaml's
`# @package habitat.dataset` directive — the package directive controls where
merged fields land in the final composed config, so `cfg.habitat.dataset.*`
still works without any caller changes).
"""

from dataclasses import dataclass

from habitat.config.default_structured_configs import DatasetConfig
from hydra.core.config_store import ConfigStore


@dataclass
class HM3DAutoVLNDatasetConfig(DatasetConfig):
    """DatasetConfig + AutoVLN-specific path fields consumed by
    `HM3DAutoVLNDatasetV1.__init__`."""

    type: str = "HM3DAutoVLN-v1"

    # Directory of `{scan}_connectivity.json` — nav graph per scan.
    nav_graph_dir: str = ""

    # Per-scan LMDB directory (recommended layout):
    #   {feature_per_scan_dir}/{split}/{scan}.lmdb
    # keys inside: `view_{vp}` -> (36, 1768) ndarray
    #              `obj_{vp}`  -> dict
    # Empty string -> fall back to legacy single-large-LMDB mode below.
    feature_per_scan_dir: str = ""

    # JSON mapping `{scan}_{vp}` -> {nbr_vp: [view_idx, dist, heading, elev]}.
    rel_angles_path: str = ""

    # Legacy single-large-LMDB mode (mutually exclusive with per-scan):
    view_feature_lmdb: str = ""
    object_feature_lmdb: str = ""

    # Optional: separate per-scan trees for view/obj (overrides combined
    # `feature_per_scan_dir` when set).
    view_feature_lmdb_dir: str = ""
    object_feature_lmdb_dir: str = ""


cs = ConfigStore.instance()
cs.store(
    package="habitat.dataset",
    group="navdsl/dataset",
    name="hm3d_autovln_schema",
    node=HM3DAutoVLNDatasetConfig,
)
