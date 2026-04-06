from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

llm = ChatOpenAI(model="gpt-4", temperature=0)
tools = [audit_nav_plan]

# 初始化具备自省能力的 Agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

instruction = "去厨房拿苏打水。注意：现在厨房门是锁着的。"
# Agent 会先生成 DSL，然后自动调用 audit_nav_plan
# 如果审计失败，它会根据 Unsat Core 的反馈重新生成计划（比如先找钥匙）
agent.run(f"请为以下任务生成 Nav-DSL 计划并进行审计：{instruction}")

class LLMNavigator:
    def __init__(self, model):
        self.model = model

    def act(self, model_input):
        response = self.model.generate(model_input)

        return self.parse_action(response)

    def parse_action(self, text):
        if "forward" in text:
            return "move_forward"
        elif "left" in text:
            return "turn_left"
        elif "right" in text:
            return "turn_right"
        else:
            return "stop"
