"""
Healthcare AI Assistant — LangGraph Agent
Pipeline: Upload Doc → OCR → Classify → Analyze → Extract → QA
"""

import os
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

from src.tools.ocr_tool import extract_text_from_file, detect_document_type
from src.tools.medical_analysis import (
    analyze_medical_report, summarize_prescription,
    extract_lab_values, answer_medical_question,
    generate_health_summary,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─── State ────────────────────────────────────────────────────────────────────

class HealthcareState(TypedDict):
    messages: Annotated[list, add_messages]
    # Document
    file_path: Optional[str]
    raw_text: Optional[str]
    doc_type: Optional[str]
    ocr_method: Optional[str]
    ocr_confidence: Optional[float]
    # Analysis
    analysis: Optional[dict]
    lab_values: Optional[dict]
    prescription_summary: Optional[dict]
    # Q&A
    conversation_history: List[dict]
    current_question: Optional[str]
    current_answer: Optional[str]
    # Settings
    language: str
    # Meta
    analyses_done: List[str]
    status: str
    errors: List[str]


# ─── Nodes ────────────────────────────────────────────────────────────────────

def ocr_node(state: HealthcareState) -> HealthcareState:
    logger.info("📄 Extracting text from document...")
    try:
        result = extract_text_from_file(state["file_path"])
        state["raw_text"] = result["text"]
        state["ocr_method"] = result["method"]
        state["ocr_confidence"] = result["confidence"]
        state["status"] = "text_extracted"
        state["messages"].append(AIMessage(
            content=f"✅ Text extracted ({result['method']}, {result['page_count']} pages, {len(result['text'].split())} words)"
        ))
    except Exception as e:
        state["errors"].append(f"OCR error: {e}")
        state["status"] = "error"
    return state


def classify_node(state: HealthcareState) -> HealthcareState:
    logger.info("🏥 Classifying document type...")
    doc_type = detect_document_type(state["raw_text"])
    state["doc_type"] = doc_type
    state["status"] = "classified"
    state["messages"].append(AIMessage(content=f"🏥 Document type: {doc_type.replace('_', ' ').title()}"))
    return state


def analyze_node(state: HealthcareState) -> HealthcareState:
    logger.info(f"🔬 Analyzing {state['doc_type']}...")
    doc_type = state["doc_type"]
    text = state["raw_text"]
    language = state.get("language", "English")

    if doc_type == "prescription":
        result = summarize_prescription(text, language)
        state["prescription_summary"] = result
        state["analyses_done"].append("prescription_summary")
    else:
        result = analyze_medical_report(text, doc_type, language)
        state["analysis"] = result
        state["analyses_done"].append("medical_analysis")

    if doc_type == "lab_report":
        lab_result = extract_lab_values(text)
        state["lab_values"] = lab_result
        state["analyses_done"].append("lab_extraction")

    state["status"] = "analyzed"
    state["messages"].append(AIMessage(content=f"✅ Analysis complete for {doc_type.replace('_', ' ').title()}"))
    return state


def qa_node(state: HealthcareState) -> HealthcareState:
    if not state.get("current_question"):
        return state

    logger.info(f"💬 Answering: {state['current_question'][:60]}...")
    context = state.get("raw_text", "")
    history = state.get("conversation_history", [])
    language = state.get("language", "English")

    answer = answer_medical_question(
        question=state["current_question"],
        context=context,
        language=language,
        conversation_history=history,
    )

    state["current_answer"] = answer
    state["conversation_history"].append({"role": "user", "content": state["current_question"]})
    state["conversation_history"].append({"role": "assistant", "content": answer})
    state["messages"].append(AIMessage(content=answer[:200] + "..."))
    state["current_question"] = None
    return state


# ─── Build Graph ──────────────────────────────────────────────────────────────

def build_healthcare_graph():
    graph = StateGraph(HealthcareState)
    graph.add_node("ocr", ocr_node)
    graph.add_node("classify", classify_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("qa", qa_node)

    graph.set_entry_point("ocr")
    graph.add_edge("ocr", "classify")
    graph.add_edge("classify", "analyze")
    graph.add_edge("analyze", "qa")
    graph.add_edge("qa", END)

    return graph.compile()


def process_document(file_path: str, language: str = "English") -> HealthcareState:
    """Process a medical document end to end"""
    graph = build_healthcare_graph()
    initial: HealthcareState = {
        "messages": [HumanMessage(content=f"Analyze medical document: {file_path}")],
        "file_path": file_path,
        "raw_text": None,
        "doc_type": None,
        "ocr_method": None,
        "ocr_confidence": None,
        "analysis": None,
        "lab_values": None,
        "prescription_summary": None,
        "conversation_history": [],
        "current_question": None,
        "current_answer": None,
        "language": language,
        "analyses_done": [],
        "status": "start",
        "errors": [],
    }
    return graph.invoke(initial)


def ask_question(state: HealthcareState, question: str) -> HealthcareState:
    """Ask a follow-up question about the document"""
    state["current_question"] = question
    state = qa_node(state)
    return state
