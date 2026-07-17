import sys
sys.path.insert(0, '.')

# 复用 navdsl/run.py 的注册 import
import navdsl.data_adapter.hm3d_autovln_dataset
import navdsl.tasks.vln_graph_nav
import navdsl.sensor.viewpoint_feature_sensor
import navdsl.sensor.object_feature_sensor
import navdsl.sensor.candidate_viewpoints_sensor
import navdsl.sensor.graph_nodes_sensor
import navdsl.measurements.viewpoint_success

# Hydra 加载实验 config
from omegaconf import OmegaConf
import hydra
from hydra import initialize_config_dir, compose
import os

from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.config.default_structured_configs import HabitatBaselinesConfigPlugin
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

class HabitatConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path):
        search_path.append(provider='habitat', path='config/')

register_hydra_plugin(HabitatConfigPlugin)
register_hydra_plugin(HabitatBaselinesConfigPlugin)

with initialize_config_dir(config_dir=os.path.abspath('config'), version_base=None):
    cfg = compose(config_name='experiments/hm3d_autovln_graph_nav',
                  overrides=[
                      f'habitat.dataset.data_path=$BASE/datasets/vln/hm3d/autovln/v1.0/DSL/{{split}}/{{split}}.json.gz',
                      f'habitat.dataset.scenes_dir=$BASE/versioned_data/hm3d-0.2/hm3d/',
                      f'habitat.dataset.scene_dataset_config=$BASE/versioned_data/hm3d-0.2/hm3d/hm3d_annotated_basis.scene_dataset_config.json',
                      f'habitat.dataset.nav_graph_dir=$BASE/datasets/vln/hm3d/autovln/v1.0/NAV_GRAPH/connectivity',
                      f'habitat.dataset.feature_per_scan_dir=$BASE/datasets/vln/hm3d/autovln/v1.0/NAV_GRAPH/features/per_scan',
                      f'habitat.dataset.rel_angles_path=$BASE/datasets/vln/hm3d/autovln/v1.0/NAV_GRAPH/annotations/scanvp_candview_relangles.json',
                      "habitat.dataset.content_scenes=['00000-kfPV7w3FaU5']",
                      'habitat.dataset.split=train',
                  ])

# Patch config with absolute pretrained weights path
cfg.habitat_baselines.il.duet.checkpoint = '$BASE/datasets/vln/hm3d/autovln/v1.0/REVERIE/expr_duet/pretrain_hm3d_v1/pseudo3d-depth2-cmt-timm.vitb16-mlm.sap.og-init.lxmert-bsz.64/ckpts/model_step_35000.pt'

from habitat.config.default import patch_config
cfg = patch_config(cfg)

print('=== Config OK ===')
print(f'trainer_name: {cfg.habitat_baselines.trainer_name}')
print(f'task type: {cfg.habitat.task.type}')
print(f'dataset type: {cfg.habitat.dataset.type}')
print(f'dataset split: {cfg.habitat.dataset.split}')

# Instantiate dataset
from navdsl.data_adapter.hm3d_autovln_dataset import HM3DAutoVLNDatasetV1
ds = HM3DAutoVLNDatasetV1(cfg.habitat.dataset)
print(f'\\n=== Dataset OK ===')
print(f'num_episodes: {len(ds.episodes)}')

ep = ds.episodes[0]
print(f'first episode: id={ep.episode_id} scan={ep.scene_scan_id}')
print(f'  start_vp={ep.start_viewpoint_id}')
print(f'  ref_path_len={len(ep.reference_viewpoints)}')

# Read one viewpoint's features
view = ds.get_view_features(ep.scene_scan_id, ep.start_viewpoint_id)
obj = ds.get_object_features(ep.scene_scan_id, ep.start_viewpoint_id)
cands = ds.get_candidates(ep.scene_scan_id, ep.start_viewpoint_id)
print(f'\\n=== Feature reads OK ===')
print(f'view_features: shape={view.shape}, norm={float((view**2).sum()**0.5):.2f}')
print(f"object_features: {len(obj['obj_ids'])} objects, fts.shape={obj['fts'].shape}")
print(f'candidates: {len(cands)} neighbors')
