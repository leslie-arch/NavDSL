import os

import numpy as np
# from magnum import shaders, text
# from magnum.platform.glfw import Application

import habitat_sim

from habitat_sim import ReplayRenderer, ReplayRendererConfiguration, physics
from habitat_sim.simulator import Configuration
from habitat_sim.sim import SimulatorConfiguration
from habitat_sim.agent import AgentConfiguration
from habitat_sim.sensor import CameraSensorSpec, SensorType
from habitat_sim.logging import LoggingContext, logger
from habitat_sim.utils.common import quat_from_angle_axis
from habitat_sim.utils.settings import default_sim_settings, make_cfg

def make_sim_config(scene_path):
    sim_settings = default_sim_settings
    # sim_cfg = make_cfg(sim_settings)
    sim_cfg = SimulatorConfiguration()
    print(f"Simulator config:: {sim_settings}")
    print(sim_cfg)

    # scenes_list = os.listdir(scene_path)
    sim_cfg.scene_id = os.path.join(scene_path, '00000-kfPV7w3FaU5', 'kfPV7w3FaU5.basic.glb')
    sim_cfg.scene_dataset_config_file = 'default'

    agent_cfg = AgentConfiguration()

    # 添加传感器
    rgb_sensor = CameraSensorSpec()
    rgb_sensor.uuid = "rgb"
    rgb_sensor.sensor_type = SensorType.COLOR
    rgb_sensor.resolution = [256, 256]

    depth_sensor = CameraSensorSpec()
    depth_sensor.uuid = "depth"

    agent_cfg.sensor_specifications = [rgb_sensor, depth_sensor]

    return Configuration(sim_cfg, [agent_cfg])


class HabitatEnv:
    def __init__(self, scene_path):
        self.sim = self._init_sim(scene_path)
        self.agent = self.sim.initialize_agent(0)

    def _init_sim(self, scene_path):
        cfg = make_sim_config(scene_path)
        return habitat_sim.Simulator(cfg)

    def reset(self, start_state):
        self.agent.set_state(start_state)
        return self.get_observation()

    def step(self, action):
        self.sim.step(action)
        return self.get_observation()

    def get_observation(self):
        return self.sim.get_sensor_observations()
