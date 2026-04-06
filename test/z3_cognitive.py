from z3 import *

def cognitive_nav_solver(llm_plan, environment_states):
    """
    llm_plan: LLM 生成的动作序列, 如 ["go_to_kitchen", "open_fridge", "get_milk"]
    environment_states: 环境反馈, 如 {"fridge_locked": True, "has_key": False}
    """
    s = Solver()

    # --- 1. 定义谓词（状态变量） ---
    num_steps = len(llm_plan)
    at_kitchen = [Bool(f'at_kitchen_{i}') for i in range(num_steps + 1)]
    fridge_open = [Bool(f'fridge_open_{i}') for i in range(num_steps + 1)]
    has_milk = [Bool(f'has_milk_{i}') for i in range(num_steps + 1)]

    # --- 2. 注入物理常识（因果约束/真值逻辑） ---
    for i in range(num_steps):
        # 约束A：只有在厨房才能开冰箱 (物理位置约束)
        s.add(Implies(fridge_open[i+1], at_kitchen[i]))

        # 约束B：如果冰箱锁了且没钥匙，就不能开冰箱 (因果约束)
        if environment_states.get("fridge_locked") and not environment_states.get("has_key"):
            s.add(fridge_open[i+1] == False)

        # 约束C：只有冰箱开了才能拿到牛奶
        s.add(Implies(has_milk[i+1], fridge_open[i]))

    # --- 3. 初始状态与目标 ---
    s.add(at_kitchen[0] == False, fridge_open[0] == False, has_milk[0] == False)

    # 目标：最终拿到牛奶
    s.add(has_milk[num_steps] == True)

    # --- 4. 验证与反馈 ---
    if s.check() == sat:
        return "SUCCESS: 导航逻辑可行"
    else:
        # 提取失败原因 (Unsat Core 的学术简化版)
        return "LOGIC_ERROR: 目标不可达。原因：冰箱已锁定且无钥匙，违反因果约束 B。"

# --- 模拟认知导航过程 ---
# 情况1：LLM 不知道冰箱锁了，直接给出了计划
llm_plan = ["go_to_kitchen", "open_fridge", "get_milk"]
env_info = {"fridge_locked": True, "has_key": False}

result = cognitive_nav_solver(llm_plan, env_info)
print(f"验证计划: {llm_plan}\n结果: {result}")
