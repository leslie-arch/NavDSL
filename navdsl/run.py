# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from pathlib import Path
from omegaconf import OmegaConf
# The following imports require habitat to be installed, and despite not being used by
# this script itself, will register several classes and make them discoverable by Hydra.
# This run.py script is expected to only be used when habitat is installed, thus they
# are hidden here instead of in an __init__.py file. This avoids import errors when used
# in an environment without habitat, such as when doing real-world deployment. noqa is
# used to suppress the unused import and unsorted import warnings by ruff.
try:
    import frontier_exploration  # noqa
except ImportError:
    # frontier_exploration is only needed for ObjectNav tasks; DUET
    # training works without it. Skipping the import lets the entry point
    # run in envs where this optional dep isn't installed.
    pass
import hydra  # noqa
from habitat import get_config  # noqa
from habitat.config import read_write  # noqa F401
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.run import execute_exp
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

from omegaconf import DictConfig

# (register sensors, observations, policies, and trainer)
# ObjectNav-only deps — wrapped in try/except so DUET training doesn't
# require the full ObjectNav dependency stack (frontier_exploration,
# depth_camera_filtering, ...). Each of these registers habitat components
# via side effects; if any is missing, the corresponding task is simply
# unavailable, but the entry point stays loadable.
_OBJECTNAV_DEPS_OK = True
try:
    import navdsl.measurements.traveled_stairs  # noqa: F401
    import navdsl.obs_transformers.resize  # noqa: F401
    from navdsl.config import objectnav_structed_config  # noqa: F401
    from navdsl.data_adapter import object_nav_hm3d_dataset  # noqa F401
    import navdsl.policy.action_replay_policy  # noqa: F401
    import navdsl.policy.habitat_policies  # noqa: F401
    import navdsl.utils  # noqa F401  (registers DSL trainers)
except ImportError as _e:
    _OBJECTNAV_DEPS_OK = False
    print(f"[navdsl.run] skipping ObjectNav deps: {_e}")

# DUET (HM3D-AutoVLN graph-nav) — always loaded, no optional deps.
import navdsl.measurements.viewpoint_success  # noqa: F401
import navdsl.config  # noqa: F401  (registers HM3DAutoVLNDatasetConfig schema)
import navdsl.data_adapter.hm3d_autovln_dataset  # noqa: F401
import navdsl.tasks.vln_graph_nav  # noqa: F401
import navdsl.sensor.viewpoint_feature_sensor  # noqa: F401
import navdsl.sensor.object_feature_sensor  # noqa: F401
import navdsl.sensor.candidate_viewpoints_sensor  # noqa: F401
import navdsl.sensor.graph_nodes_sensor  # noqa: F401
import navdsl.policy.duet.duet_policy  # noqa: F401
import navdsl.utils.duet_trainer  # noqa: F401
# import navdsl.sensor.symbolic_fact_sensor  # noqa: F401


class HabitatConfigPlugin(SearchPathPlugin):
    """Add config to habitat search path."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="habitat", path="config/")


register_hydra_plugin(HabitatConfigPlugin)


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="experiments/dsl_objectnav_hm3d",
)
def main(cfg: DictConfig) -> None:
    assert os.path.isdir("data"), "Missing 'data/' directory!"
    trainer_name = cfg.habitat_baselines.trainer_name

    # Skip DSL-specific preflight for non-DSL trainers
    if trainer_name in ("dsl",):
        if not os.path.isfile("data/dummy_policy.pth"):
            print("Dummy policy weights not found! Please run the following command first:")
            print("python -m navdsl.utils.generate_dummy_policy")
            exit(1)

    cfg = patch_config(cfg)

    # Save the resolved config next to the checkpoint (DSL-specific path skipped
    # for non-DSL trainers that don't define rl.policy.main_agent.pointnav_policy_path).
    if trainer_name in ("dsl",):
        try:
            config_save_path = Path(
                cfg.habitat_baselines.rl.policy.main_agent.pointnav_policy_path
            )
            config_save_fp = config_save_path.with_suffix('.yaml')
            with open(config_save_fp, "w+") as f:
                OmegaConf.save(config=cfg, f=f)
        except Exception:
            pass

    execute_exp(cfg, "eval" if cfg.habitat_baselines.evaluate else "train")


if __name__ == "__main__":
    main()
