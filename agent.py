from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from vector_store import load_vectorstore

load_dotenv()

llm = ChatOllama(model="mistral", temperature=0)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    session_id: str

active_sessions = {}

def build_agent_for_session(session_id):
    vector_store_path = f"sessions/{session_id}/vector_store"
    vectorstore = load_vectorstore(vector_store_path)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # @tool
    # def retriever_tool(query: str) -> str:
    #     docs = retriever.invoke(query)
    #     if not docs:
    #         return "No relevant information found."
    #     return "\n\n".join([doc.page_content for doc in docs])
    @tool(description="Retrieves relevant information from the uploaded documents based on the query.")
    def retriever_tool(query: str) -> str:
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found."
        return "\n\n".join([doc.page_content for doc in docs])

    tools = [retriever_tool]
    local_llm = llm.bind_tools(tools)

    def should_continue(state: AgentState):
        last_message = state['messages'][-1]
        return hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0

    system_prompt = "You are DocuMentor. Use the retriever tool to answer from the provided documents. Do not hallucinate."

    tools_dict = {tool.name: tool for tool in tools}

    def call_llm(state: AgentState) -> AgentState:
        messages = [SystemMessage(content=system_prompt)] + list(state['messages'])
        response = local_llm.invoke(messages)
        return {'messages': [response], 'session_id': state['session_id']}

    def take_action(state: AgentState) -> AgentState:
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            tool_name = t['name']
            query = t['args'].get('query', '')
            result = tools_dict[tool_name].invoke(query) if tool_name in tools_dict else "Invalid tool."
            results.append(ToolMessage(tool_call_id=t['id'], name=tool_name, content=str(result)))
        return {'messages': results, 'session_id': state['session_id']}

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever_agent", take_action)
    graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
    graph.add_edge("retriever_agent", "llm")
    graph.set_entry_point("llm")

    return graph.compile()

def get_or_create_agent(session_id):
    if session_id not in active_sessions:
        active_sessions[session_id] = build_agent_for_session(session_id)
    return active_sessions[session_id]
