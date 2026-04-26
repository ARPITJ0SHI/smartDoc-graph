"""LangGraph StateGraph definition for the RAG pipeline.

Graph Flow:
  START → load_memory → hyde_generation → retrieve_faiss → crag_evaluation
  crag_evaluation → (conditional):
    - fallback=True  → fallback_node → END
    - fallback=False → rerank → generate_answer → update_memory → END

Short-term memory is managed by LangGraph's MemorySaver checkpointer.
Messages accumulate automatically across invocations for the same thread_id.
"""

from typing import Annotated, Dict, List

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver

from app.workflows.nodes import (
    load_memory_node,
    hyde_generation_node,
    retrieve_faiss_node,
    crag_evaluation_node,
    rerank_node,
    generate_answer_node,
    update_memory_node,
    fallback_node,
)


class RAGState(TypedDict, total=False):
    """State schema for the RAG pipeline graph."""
    # Input
    query: str
    session_id: str  # UUID as string for serialization

    # Memory — messages accumulate via add_messages reducer
    messages: Annotated[list, add_messages]
    conversation_context: str  # formatted string for prompts

    # Retrieval
    hyde_doc: str
    retrieved_chunks: List[Dict]
    reranked_chunks: List[Dict]

    # CRAG
    relevant_count: int
    fallback: bool

    # Output
    final_answer: str
    sources: List[Dict]


def _route_after_crag(state: RAGState) -> str:
    """Conditional edge: route to fallback or re-rank based on CRAG result."""
    if state.get("fallback", False):
        return "fallback_node"
    return "rerank"


def build_rag_graph() -> StateGraph:
    """Construct and compile the RAG workflow graph."""
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("load_memory", load_memory_node)
    workflow.add_node("hyde_generation", hyde_generation_node)
    workflow.add_node("retrieve_faiss", retrieve_faiss_node)
    workflow.add_node("crag_evaluation", crag_evaluation_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("update_memory", update_memory_node)
    workflow.add_node("fallback_node", fallback_node)

    # Define edges
    workflow.add_edge(START, "load_memory")
    workflow.add_edge("load_memory", "hyde_generation")
    workflow.add_edge("hyde_generation", "retrieve_faiss")
    workflow.add_edge("retrieve_faiss", "crag_evaluation")

    # Conditional edge after CRAG
    workflow.add_conditional_edges(
        "crag_evaluation",
        _route_after_crag,
        {
            "fallback_node": "fallback_node",
            "rerank": "rerank",
        },
    )

    # Continue from rerank → generate → update memory → END
    workflow.add_edge("rerank", "generate_answer")
    workflow.add_edge("generate_answer", "update_memory")
    workflow.add_edge("update_memory", END)

    # Fallback → END
    workflow.add_edge("fallback_node", END)

    # Compile with MemorySaver checkpointer for short-term memory
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# Compiled graph singleton
rag_app = build_rag_graph()
