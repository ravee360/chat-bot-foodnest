from langgraph.graph import StateGraph, add_messages, MessageGraph, END
from backend.app.services.agent_chain import generation_chain, reflection_chain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from typing import TypedDict, Optional

class BasicState(TypedDict):
    message : Optional[str]
    svg : Optional[str]
    improvement : Optional[str]

from backend.app import GROQ_API_KEY, GROQ_MODEL

# state_graph = StateGraph(BasicState)
graph = MessageGraph()

REFLECT = "reflect"
GENERATE = "generate"

def generate(message: str):
    return generation_chain.invoke({
        "message" : message
    })

def reflect(message: str):
    return reflection_chain.invoke({
        "message": message
    })

def should_continue(state):
    if(len(state)> 6):
        return END
    return "generate"

graph.add_node(GENERATE, generate)
graph.add_node(REFLECT, reflect)
graph.set_entry_point("generate")

graph.add_edge(GENERATE, REFLECT)
graph.add_conditional_edges(GENERATE, should_continue)

app = graph.compile()

response = app.invoke({"message": " A: 40 , B:30, C: 60"})
print(response)