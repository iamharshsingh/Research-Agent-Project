# Step 1: LangGraph Setup with Hugging Face Dual Agents
from langgraph.graph import StateGraph, END
from langchain_community.llms import HuggingFaceHub
from langchain_core.runnables import RunnableLambda
from Tools.Travely import get_tavily_tool
from langchain.agents import initialize_agent, AgentType
from typing import TypedDict
import os
from dotenv import load_dotenv

load_dotenv()

# Step 2: Define shared state
class AgentState(TypedDict):
    question: str
    research_result: str
    drafted_answer: str

# Step 3: Research Agent using Hugging Face + Tavily
def create_research_agent():
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-xl",  # You can use other models too
        model_kwargs={"temperature": 0.3, "max_length": 512}
    )
    tools = [get_tavily_tool()]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    def research_fn(state: AgentState):
        result = agent.invoke({"input": state["question"]})
        return {"research_result": result["output"]}

    return RunnableLambda(research_fn)

# Step 4: Drafting Agent using Hugging Face
def create_drafting_agent():
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-xl",
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

    def draft_fn(state: AgentState):
        prompt = (
            f"Write a well-structured response to the following question using the research:\n\n"
            f"Question: {state['question']}\n\n"
            f"Research:\n{state['research_result']}"
        )
        answer = llm.invoke(prompt)
        return {"drafted_answer": answer}

    return RunnableLambda(draft_fn)

# Step 5: Build the LangGraph Flow
def build_graph():
    research_agent = create_research_agent()
    drafting_agent = create_drafting_agent()

    graph = StateGraph(AgentState)
    graph.add_node("research", research_agent)
    graph.add_node("draft", drafting_agent)
    graph.set_entry_point("research")
    graph.add_edge("research", "draft")
    graph.add_edge("draft", END)

    return graph.compile()

# Example usage
if __name__ == "__main__":
    graph_app = build_graph()
    input_state = {"question": "What are the benefits of agentic AI systems in research workflows?"}
    result = graph_app.invoke(input_state)
    print(result["drafted_answer"])
