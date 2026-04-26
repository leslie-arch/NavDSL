"""Symbolic Fact Sensor."""

from typing import (
    Any,
    Dict,
    Union,
    List,
    Optional
)
import abc

import attr
import numpy as np
from gym import Space, spaces
from omegaconf import DictConfig

import habitat  # noqa: F401
import habitat_sim
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, Sensor, SensorTypes, VisualObservation
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
)
from habitat.tasks.nav.object_nav_task import (
    ObjectGoalNavEpisode,
    ObjectViewLocation,
    ObjectGoal,
)

# from habitat.sims.habitat_simulator.habitat_simulator import HabitatSimSensor

# 在 Habitat 中，传感器主要分为两类路径，它们通过是否继承 HabitatSimSensor 产生分化：
# 路径 A：底层模拟器传感器 (Sim-bound Sensors)
# 代表：HabitatSimRGBSensor, HabitatSimDepthSensor, HabitatSimSemanticSensor。
# 继承链：class HabitatSimRGBSensor(RGBSensor, HabitatSimSensor)。
# 关键点：HabitatSimSensor 是一个 Mixin。
# 在 HabitatSim 模拟器初始化时，源码中有一行判断：isinstance(sensor, HabitatSimSensor)。
# 结果：只有满足这个 isinstance 判断的传感器，才会进入 create_sim_config 逻辑，
# 进而触发对 observation_space.shape[:2] 的强制检查。
# 路径 B：任务级/逻辑传感器 (Task-level Sensors)
# 代表：ObjectGoalSensor, GPSSensor, CompassSensor。
# 继承链：直接继承自 habitat.core.simulator.Sensor
# 结果：由于它们不是 HabitatSimSensor 的子类，HabitatSim 的初始化逻辑会跳过它们，
# 不会尝试去同步 C++ 端的 resolution。

class SymbolicFactSensor(Sensor, metaclass=abc.ABCMeta):
    """Symbolic Fact Sensor Mixiin."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args, **kwargs) -> str:
        return "symbolic_facts"

    def _get_sensor_type(self, *args, **kwargs) -> SensorTypes:
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        raise NotImplementedError

    def get_observation(self, *args: Any, **kwargs: Any) -> VisualObservation:
        raise NotImplementedError


@registry.register_sensor
# class HabitatSimSymbolicFactSensor(SymbolicFactSensor, HabitatSimSensor):
class HabitatSymbolicFactSensor(Sensor):
    """把 Habitat 的底层 3D 语义数据，抽离为 Z3 能看懂的'逻辑事实'."""

    # _get_default_spec = habitat_sim.SemanticSensorSpec
    sim_sensor_type = habitat_sim.SensorType.SEMANTIC

    cls_uuid: str = "symbolic_fact"

    def __init__(self,
                 sim,
                 config: DictConfig,
                 dataset: "ObjectNavDatasetV1",
                 *args: Any,
                 **kwargs: Any) -> None:
        self._sim = sim
        self._dataset = dataset
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args, **kwargs) -> SensorTypes:
        return SensorTypes.SEMANTIC

    def _get_default_spec(self):
        return habitat_sim.SemanticSensorSpec()

    def _get_current_room(self, agent_pos, episode: ObjectGoalNavEpisode):
        """利用HM3D的Region属性快速定位当前房间."""
        # print(f"_get_current_room: episode {episode.goals_key}")
        # 1. 检查语义场景是否真的载入了元数据
        if self._sim.semantic_scene and len(self._sim.semantic_scene.objects) > 0:
            print(f"Number of Objects: {len(self._sim.semantic_scene.objects)}\n"
                  f"Graph node: {self._sim.get_active_scene_graph().get_root_node().object_semantic_id}")
        print(f"_get_current_room: dataset scenes_path {self._dataset.content_scenes_path}\n"
              f"episodes: length {len(self._dataset.episodes)} scene_dataset_config: {self._dataset.episodes[1].scene_dataset_config}")
        print(f"_get_current_room: episode id {episode.episode_id} scene_id {episode.scene_id}, "
               f"scene dataset config {episode.scene_dataset_config}, "
               f"info: {episode.info}, initialize template: {self._sim.get_stage_initialization_template()}")

        for obj in self._sim.semantic_scene.objects[:10]:
            print(f"object --- [{obj.semantic_id}.{obj.id}]: {obj.category.name()}")
        #     print(f"Number of Regions: {len(self._sim.semantic_scene.regions)}")
        #     print(f"_get_current_room: semantic_scene {self._sim.semantic_scene}")
        regions = self._sim.semantic_scene.regions
        if not len(regions) == 0:
            print(f"_get_current_room: regions count: {len(regions)}")
        current_region_name = "unknown_area"
        for region in regions:
            if region.aabb.contains(agent_pos):
                current_region_name = region.category.name()

        return current_region_name

    def _get_visible_objects(self, observations, max_dist=5.0):
        """
        利用 Semantic Observation 提取当前视野内的物体.

        比遍历全场景物体更符合“具身感知”
        """
        semantic_obs = observations["semantic"]
        unique_obj_ids = np.unique(semantic_obs)
        if not np.all(semantic_obs == 0):
            print(f"_get_visible_objects: find {len(semantic_obs)} visible object(s).")

        visible_facts = []
        for obj_id in unique_obj_ids:
            # 过滤背景 (ID 0 通常是背景)
            if obj_id == 0:
                continue

            # 获取物体元数据
            obj = self._sim.semantic_scene.objects[obj_id]
            print(f"_get_visible_objects: ID {obj_id} {obj} at {obj.aabb.center}/{self._sim.get_agent_state().position}")
            # 距离过滤
            dist = np.linalg.norm(obj.aabb.center -
                                  self._sim.get_agent_state().position)

            if dist < max_dist:
                visible_facts.append({
                    "id": obj.id,
                    "class": obj.category.name(),
                    "pos": obj.aabb.center.tolist()
                })
        return visible_facts

    def _get_door_status(self, visible_objects):
        """
        在 HM3D 中，门通常是静态物体.

        如果是交互式任务（如 Habitat-Rearrange），可以查询其打开角度。
        如果是静态 ObjectNav，通常根据其类别名判定是否为门。
        """
        doors = [obj for obj in visible_objects if "door" in obj["class"]]
        # 示例：通过物体状态或简单的逻辑模拟门锁
        # 在实际实验中，可以根据 Episode 的元数据预设某些门是 Locked
        return [{"id": d["id"], "is_locked": False} for d in doors]

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        sensor_shape = (1,)
        max_value = self.config.goal_spec_max_val - 1
        if self.config.goal_spec == "TASK_CATEGORY_ID":
            max_value = max(
                self._dataset.category_to_task_category_id.values()
            )

        return spaces.Box(
            low=0, high=max_value, shape=sensor_shape, dtype=np.int64
        )

    # observations 是一个正在构建中的观测字典。
    # 从 Simulator（模拟器层） 传递到 EmbodiedTask（任务层）
    # 所有底层传感器的原始数据（如 rgb, depth, semantic）封装进一个初始的 observations 字典
    # 遍历所有注册的 lab_sensors（Task Sensors）,任务层会把这个已经包含模拟器数据的字典作为参数
    def get_observation(
        self,
        observations,
        *args: Any,
        episode: ObjectGoalNavEpisode,
        **kwargs: Any,
    ) -> Optional[np.ndarray]:
        """读取当前智能体位置及可见物体，转化为 Nav-ASL 事实."""
        agent_state = self._sim.get_agent_state()
        current_room = self._get_current_room(agent_state.position, episode)
        visible_objects = self._get_visible_objects(observations)

        facts = [f"At(robot, {current_room})"]
        for obj in visible_objects:
            facts.append(f"Visible({obj.name})")

        door_stats = self._get_door_status(visible_objects)
        for door in door_stats:
            if door["is_locked"]:
                facts.append(f"IsLocked(door_{door['id']})")

        category_name = episode.object_category
        if len(episode.goals) == 0:
            logger.error(
                f"No goal specified for episode {episode.episode_id}."
            )
            raise RuntimeError(
                "SymbolicFactSensor: Empty goals ObjectGoalSensor."
            )
        if not isinstance(episode.goals[0], ObjectGoal):
            logger.error(
                f"SymbolicFactSensor: First goal should be ObjectGoal, episode {episode.episode_id}."
            )
            raise RuntimeError(
                "SymbolicFactSensor: Wrong goals ObjectGoalSensor."
            )
        if self.config.goal_spec == "TASK_CATEGORY_ID":
            return np.array(
                [self._dataset.category_to_task_category_id[category_name]],
                dtype=np.int64,
            )
        elif self.config.goal_spec == "OBJECT_ID":
            obj_goal = episode.goals[0]
            assert isinstance(obj_goal, ObjectGoal)  # for type checking
            return np.array([obj_goal.object_name_id], dtype=np.int64)
        else:
            raise RuntimeError(
                "Wrong goal_spec specified for ObjectGoalSensor."
            )

        return np.random.randint(1, 11, 1)

    # def get_observation(self,
    #                     sim_obs: Dict[str, Union[np.ndarray, bool, "Tensor"]]  # noqa: F821
    #                     ) -> VisualObservation:
    #
    #     import json
    #     print(f"get_observation sim_obs:\n{json.dumps(sim_obs)}")
    #
    #     return []
    #     """读取当前智能体位置及可见物体，转化为 Nav-ASL 事实."""
    #     agent_state = self._sim.get_agent_state()
    #     # 1. 伪代码：获取当前房间
    #     current_room = self._get_current_room(agent_state.position)
    #
    #     # 2. 伪代码：获取视野内可见物体（通过 Habitat 的 Semantic Scene）
    #     visible_objects = self._get_visible_objects()
    #
    #     # 3. 构造返回给 LLM 的事实列表
    #     facts = [f"At(robot, {current_room})"]
    #     for obj in visible_objects:
    #         facts.append(f"Visible({obj.name})")
    #
    #     # 4. 门的状态（逻辑示例）
    #     door_stats = self._get_door_status(visible_objects)
    #     for door in door_stats:
    #         if door["is_locked"]:
    #             facts.append(f"IsLocked(door_{door['id']})")
    #
    #     return np.random.randint(1, 11, 1)
