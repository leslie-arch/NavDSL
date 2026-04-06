import numpy as np
from z3 import *
from langchain.tools import tool
from typing import List, Dict, Any

class NavDSLValidator:
    def __init__(self):
        self.solver = Solver()
        self.solver.set("timeout", 1000) # 1秒超时，适合实时导航

        # --- 定义 Nav-DSL 基础谓词 ---
        self.Object = DeclareSort('Object')
        self.Room = DeclareSort('Room')
        self.Robot = Const('robot', self.Object)

        # 谓词：At(entity, location)
        self.At = Function('At', self.Object, self.Room, BoolSort())
        # 谓词：Holding(robot, object)
        self.Holding = Function('Holding', self.Object, self.Object, BoolSort())
        # 谓词：IsOpened(door)
        self.IsOpened = Function('IsOpened', self.Object, BoolSort())

        # 物理公理：如果要进入房间，门必须是开的
        # 这里仅为示例，实际可从你的 Axiom 库加载
        self.door_of = Function('door_of', self.Room, self.Object)

    def reset_with_context(self, facts: List[str]):
        """
        根据 Habitat 传感器的感知结果初始化 Z3 环境事实
        """
        self.solver.reset()
        # 示例：将感知到的 "door_1 is closed" 转化为 Z3 语句
        # 实际开发中，这里需要一个简单的映射字典
        for fact in facts:
            if "door_1_closed" in fact:
                self.solver.add(Not(self.IsOpened(Const('door_1', self.Object))))
            if "robot_at_hallway" in fact:
                self.solver.add(self.At(self.Robot, Const('hallway', self.Room)))

    def verify_plan(self, dsl_steps: List[str]) -> (bool, str):
        """
        审计 DSL 步骤的逻辑一致性
        """
        # 为每条指令创建命名的断言，以便提取 Unsat Core
        for i, step in enumerate(dsl_steps):
            label = f"step_{i}"

            # 简单解析示例：DSL "navigate_to(kitchen)"
            if "navigate_to" in step:
                target_room = Const('kitchen', self.Room)
                target_door = self.door_of(target_room)

                # 核心审计逻辑：进入房间需要门是开的
                condition = self.IsOpened(target_door)
                # 使用带有标签的断言，方便追踪冲突
                self.solver.assert_and_track(condition, label)

        if self.solver.check() == sat:
            return True, "Success"
        else:
            # 提取导致冲突的步骤 (Unsat Core)
            core = self.solver.unsat_core()
            feedback = f"Logical conflict detected in: {core}. Physical invariant violated."
            return False, feedback


class ExtendedNavValidator:
    def __init__(self):
        self.solver = Solver()

        # 定义基本类型
        Entity = DeclareSort('Entity')
        Location = DeclareSort('Location')

        # 定义常量
        robot = Const('robot', Entity)

        # 定义谓词
        At = Function('At', Entity, Location, BoolSort())
        Holding = Function('Holding', Entity, Entity, BoolSort())
        IsOpened = Function('IsOpened', Entity, BoolSort())
        HasDoor = Function('HasDoor', Location, Location, Entity, BoolSort())
        HandFree = Function('HandFree', Entity, BoolSort())

        # --- 注入全局公理 ---

        # 1. 门禁公理：跨越房间必须门是开的
        r1, r2, d = Consts('r1 r2 d', Location), Const('d', Entity)
        self.solver.add(ForAll([r1, r2, d],
            Implies(HasDoor(r1, r2, d),
                    Implies(At(robot, r2), IsOpened(d))) # 简化逻辑：在r2意味着过了门d
        ))

        # 2. 抓取公理：抓取前必须HandFree
        obj = Const('obj', Entity)
        self.solver.add(ForAll([obj],
            Implies(Holding(robot, obj), Not(HandFree(robot)))
        ))

    def add_current_observation(self, obs_facts):
        """
        从 Habitat 传感器获取的实时事实
        """
        # 例如：发现厨房门是关着的
        kitchen_door = Const('kitchen_door', DeclareSort('Entity'))
        self.solver.add(IsOpened(kitchen_door) == False)


# --- 封装为 LangChain Tool ---
@tool
def audit_nav_plan(plan_str: str) -> str:
    """
    使用 Z3 形式化验证器审计导航计划。
    输入应为 Nav-DSL 指令列表（如 ['navigate_to(kitchen)', 'pick(apple)']）。
    如果返回 'Success'，则计划可执行；否则返回具体的冲突原因。
    """
    validator = NavDSLValidator()
    # 模拟从感知层获取的事实
    current_facts = ["door_1_closed", "robot_at_hallway"]
    validator.reset_with_context(current_facts)

    plan_steps = plan_str.strip("[]").replace("'", "").split(", ")
    is_valid, message = validator.verify_plan(plan_steps)

    return message
