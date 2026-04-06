from z3 import *
import json

class HM3DLogicalEnvironment:
    def __init__(self, scene_metadata_path):
        self.solver = Solver()
        self.solver.set(unsat_core=True)

        # 1. 定义 Nav-DSL 核心类型
        self.Entity = DeclareSort('Entity')
        self.Room = DeclareSort('Room')

        # 2. 定义核心谓词
        self.At = Function('At', self.Entity, self.Room, BoolSort())
        self.IsLocked = Function('IsLocked', self.Entity, BoolSort())
        self.Connects = Function('Connects', self.Entity, self.Room, self.Room, BoolSort())

        # 加载 HM3D 场景元数据（假设已预处理为 JSON）
        with open(scene_metadata_path, 'r') as f:
            self.scene_data = json.load(f)

    def initialize_scene_logic(self):
        """
        将 HM3D 场景中的拓扑结构映射为 Z3 事实
        """
        # 示例：定义房间
        kitchen = Const('kitchen', self.Room)
        hallway = Const('hallway', self.Room)
        robot = Const('robot', self.Entity)
        k_door = Const('k_door', self.Entity)

        # 注入场景事实：门 k_door 连接走廊和厨房
        self.solver.add(self.Connects(k_door, hallway, kitchen))

        # 注入动态观测：当前门是锁着的
        # 为每条事实添加标签，以便在冲突时回溯
        self.solver.assert_and_track(self.IsLocked(k_door), "fact_door_locked")
        self.solver.assert_and_track(self.At(robot, hallway), "fact_robot_location")

    def audit_dsl_action(self, action_type, target_entity, target_room):
        """
        审计单条 DSL 动作是否违反物理公理
        """
        # 公理：如果动作是穿越门，则该门不能是 Locked 状态
        if action_type == "cross_door":
            door = Const(target_entity, self.Entity)
            # 逻辑约束：必须满足 (NOT IsLocked(door))
            constraint = Not(self.IsLocked(door))

            # 检查一致性
            self.solver.push()
            self.solver.assert_and_track(constraint, "check_access_control")

            result = self.solver.check()
            if result == unsat:
                core = self.solver.unsat_core()
                self.solver.pop()
                return False, f"Logic Conflict: {core}"

            self.solver.pop()
            return True, "Safe"
