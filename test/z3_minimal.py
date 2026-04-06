import openai
from z3 import *

# --- 第一步：定义 Z3 环境模型（这是你论文的“硬约束”部分） ---
def check_plan_logic(actions):
    s = Solver()

    # 定义状态变量：拿杯子(holding_cup), 倒水(poured)
    # 使用 Bool 表示每个步骤后的状态
    steps = len(actions)
    holding = [Bool(f'holding_{i}') for i in range(steps + 1)]
    poured = [Bool(f'poured_{i}') for i in range(steps + 1)]

    print(f'holding: {holding}')
    print(f'poured: {poured}')

    # 初始状态：没拿杯子，没倒水
    s.add(z3.Not(holding[0]), z3.Not(poured[0]))

    # 动作逻辑约束
    for i, action in enumerate(actions):
        if action == "pick_up_cup":
            # 动作：拿杯子 -> 结果：拿着杯子，倒水状态不变
            s.add(holding[i+1] == True)
            s.add(poured[i+1] == poured[i])
        elif action == "pour_water":
            # 关键约束：倒水的前提是必须正拿着杯子
            s.add(Implies(poured[i+1], holding[i]))
            s.add(poured[i+1] == True)
            s.add(holding[i+1] == holding[i])
        else:
            return "Unknown Action"

    # 目标：最后必须完成了倒水
    s.add(poured[steps] == True)

    if s.check() == sat:
        return "PASS: 逻辑合法"
    else:
        return "FAIL: 逻辑错误（违反约束：倒水前必须先抓取杯子）"

# --- 第二步：LLM 充当翻译器（将自然语言转为动作序列） ---
def llm_translator(user_input):
    # 这里模拟 LLM 的输出。在论文中，你会展示 Prompt 工程的结果
    # 假设用户说“直接倒水”，LLM 偷懒只生成了一个动作
    if "直接" in user_input:
        return ["pour_water"]
    else:
        return ["pick_up_cup", "pour_water"]

# --- 第三步：闭环测试 ---
user_cmd = "帮我直接倒水"
plan = llm_translator(user_cmd)
result = check_plan_logic(plan)

print(f"用户指令: {user_cmd}")
print(f"LLM 生成计划: {plan}")
print(f"Z3 验证结果: {result}")
