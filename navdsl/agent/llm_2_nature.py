from z3 import *

def get_feedback_from_unsat(solver, mapping_dict):
    """
    从 Z3 求解器中提取 Unsat Core，并根据字典映射为自然语言描述。
    """
    if solver.check() == unsat:
        # 获取导致冲突的最小约束集（Labels）
        core = solver.unsat_core()
        explanations = []
        for label in core:
            # 从字典中查找该 Label 对应的自然语言描述
            msg = mapping_dict.get(str(label), f"未知约束冲突: {label}")
            explanations.append(f"- {msg}")

        feedback = "检测到逻辑冲突，请修正计划：\n" + "\n".join(explanations)
        return feedback
    else:
        return "逻辑通过，无需修正。"

# --- 实验场景：机器人倒水任务 ---
s = Solver()

# 1. 定义状态变量（时间步 0 和 1）
holding_cup_0 = Bool('holding_cup_0')
poured_1 = Bool('poured_1')

# 2. 定义【自然语言映射字典】（这是 Methodology 中的核心资产）
# Key 是 Z3 的约束标签，Value 是对应的反馈话术
EXPLANATION_MAP = {
    "C_PREC_POUR": "倒水动作执行失败，因为此时机器人手中没有杯子。请在倒水前添加 Pick_up 动作。",
    "C_INIT_STATE": "初始状态设定为机器人未持有任何物品。",
    "C_GOAL": "任务目标要求完成倒水。"
}

# 3. 添加带【标签】的约束
# 初始状态：没拿杯子
s.assert_and_track(Not(holding_cup_0), "C_INIT_STATE")

# 目标：要完成倒水
s.assert_and_track(poured_1 == True, "C_GOAL")

# 物理逻辑约束：只有在 t=0 时拿着杯子，t=1 才能倒水 (Pre-condition)
s.assert_and_track(Implies(poured_1, holding_cup_0), "C_PREC_POUR")

# --- 4. 自动化纠错演示 ---
print("--- 启动逻辑验证 ---")
feedback = get_feedback_from_unsat(s, EXPLANATION_MAP)
print(feedback)

# --- 5. 模拟 LLM 接收反馈后的修正过程 ---
if s.check() == unsat:
    print("\n--- 系统已将上述反馈发送给 LLM，正在生成修正方案... ---")
    # 模拟修正动作：插入动作使 holding_cup_0 变为 True
    s_new = Solver()
    # 重新添加约束，但假设我们添加了一个 Pick_up 动作使得逻辑闭环
    s_new.add(holding_cup_0 == True) # 修正后的逻辑：已经拿到了杯子
    s_new.add(poured_1 == True)
    s_new.add(Implies(poured_1, holding_cup_0))

    if s_new.check() == sat:
        print("修正成功：新的计划符合物理逻辑。")
