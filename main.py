import dotenv
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from tools import (
    multiply,
    add,
    subtract,
    divide,
    exponent,
    symbolic_derivative,
    symbolic_integral,
    definite_integral,
    web_searcher,
)


dotenv.load_dotenv()


llm = ChatOpenAI()
tools = [
    multiply,
    add,
    subtract,
    divide,
    exponent,
    symbolic_derivative,
    symbolic_integral,
    definite_integral,
    web_searcher,
]
prompt = hub.pull("hwchase17/openai-tools-agent")

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "Integrate x^2"})
