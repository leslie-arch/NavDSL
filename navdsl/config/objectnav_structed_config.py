# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass, fields
from typing import Tuple, List
from habitat_baselines.config.default_structured_configs import (
    PolicyConfig,
)
from habitat.config.default_structured_configs import (
    # SimulatorCameraSensorConfig
    LabSensorConfig
)
from hydra.core.config_store import ConfigStore

from ..sensor import symbolic_fact_sensor  # noqa F401

@dataclass
class ObjectNavBaseConfig:
    """
    视觉语言寻找模型的配置类.

    包含导航策略所需的各种参数
    """

    name: str = "HabitatITMPolicy"  # 策略名称
    text_prompt: str = "Seems like there is a target_object ahead."  # 文本提示
    pointnav_policy_path: str = "data/pointnav_weights.pth"  # 点导航策略路径
    depth_image_shape: Tuple[int, int] = (224, 224)  # 深度图像形状
    pointnav_stop_radius: float = 0.9  # 点导航停止半径（米）
    use_max_confidence: bool = False  # 是否使用最大置信度
    object_map_erosion_size: int = 5  # 对象地图腐蚀大小
    exploration_thresh: float = 0.0  # 探索阈值
    obstacle_map_area_threshold: float = 1.5  # 障碍地图面积阈值（平方米）
    min_obstacle_height: float = 0.61  # 最小障碍物高度
    max_obstacle_height: float = 0.88  # 最大障碍物高度
    hole_area_thresh: int = 100000  # 洞区域阈值
    use_vqa: bool = False  # 是否使用视觉问答
    vqa_prompt: str = "Is this "  # 视觉问答提示
    coco_threshold: float = 0.8  # COCO类别阈值
    non_coco_threshold: float = 0.4  # 非COCO类别阈值
    agent_radius: float = 0.18  # 代理半径

    @classmethod  # type: ignore
    @property
    def kwaarg_names(cls) -> List[str]:
        """
        返回所有配置字段名称.

        除了name字段
        用于自动构建关键字参数
        """
        # 返回除"name"外的所有字段名
        return [f.name for f in fields(DSLConfig) if f.name != "name"]

@dataclass
class DSLConfig(ObjectNavBaseConfig):
    pass


@dataclass
class VLFMConfig(ObjectNavBaseConfig):
    pass


@dataclass
class DSLPolicyConfig(DSLConfig, PolicyConfig):
    """
    视觉语言寻找模型策略配置.

    继承DSLConfig和PolicyConfig，结合两者的配置项
    """

    pass


@dataclass
class VLFMPolicyConfig(VLFMConfig, PolicyConfig):
    """
    视觉语言寻找模型策略配置
    继承VLFMConfig和PolicyConfig，结合两者的配置项
    """
    pass


@dataclass
class HabitatSymbolicFactSensorConfig(LabSensorConfig):
    r"""
    For Object Navigation tasks only. Generates a discrete observation containing
    the id of the goal object for the episode.

    :property goal_spec: A string that can take the value TASK_CATEGORY_ID or OBJECT_ID. If the value is TASK_CATEGORY_ID, then the observation will be the id of the `episode.object_category` attribute, if the value is OBJECT_ID, then the observation will be the id of the first goal object.
    :property goal_spec_max_val: If the `goal_spec` is OBJECT_ID, then `goal_spec_max_val` is the total number of different objects that can be goals. Note that this value must be greater than the largest episode goal category id.
    """
    type: str = "HabitatSymbolicFactSensor"
    goal_spec: str = "TASK_CATEGORY_ID"
    goal_spec_max_val: int = 50


# 获取配置存储实例
cs = ConfigStore.instance()

# 注册语义传感器
cs.store(package="habitat.task.lab_sensors.symbolic_fact_sensor",
         group="habitat/task/lab_sensors",
         name="symbolic_fact_sensor",
         node=HabitatSymbolicFactSensorConfig)

# 注册基础VLFM配置及策略
cs.store(group="policy", name="vlfm_config_base", node=VLFMConfig())
cs.store(group="habitat_baselines/rl/policy",
         name="vlfm_policy",
         node=VLFMPolicyConfig)

# 注册基础DSL配置及策略
cs.store(group="policy", name="dsl_config_base", node=DSLConfig())
cs.store(group="habitat_baselines/rl/policy",
         name="dsl_policy",
         node=DSLPolicyConfig)

