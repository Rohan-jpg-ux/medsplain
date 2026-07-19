"""
Medical Analysis Tool
Uses LLM to analyze medical documents, explain findings, and answer questions
IMPORTANT: Always includes medical disclaimer — AI for information only
"""

import os
import json
from typing import Optional, List
from src.utils.logger import get_logger

logger = get_logger(__name__)

DISCLAIMER = "\n\n⚕️ **Medical Disclaimer:** This AI analysis is for informational purposes only and does not constitute medical advice. Always consult a qualified healthcare professional for medical decisions."

SUPPORTED_LANGUAGES = {
    "English": "en",
    "Hindi (हिंदी)": "hi",
    "Spanish (Español)": "es",
    "French (Français)": "fr",
    "Arabic (العربية)": "ar",
    "Chinese (中文)": "zh",
    "Portuguese (Português)": "pt",
    "German (Deutsch)": "de",
}


def get_llm(temperature=0.1, max_tokens=3000):
    groq_key = os.getenv("GROQ_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if anthropic_key:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-3-5-haiku-20241022",
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=anthropic_key,
        )
    elif groq_key:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=groq_key,
        )
    else:
        raise ValueError("Set ANTHROPIC_API_KEY or GROQ_API_KEY")


def call_llm(system: str, user: str, temperature=0.1) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage
    llm = get_llm(temperature=temperature)
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return response.content.strip()


def call_llm_json(system: str, user: str) -> dict:
    llm = get_llm()
    from langchain_core.messages import HumanMessage, SystemMessage
    response = llm.invoke([
        SystemMessage(content=system + "\n\nRESPOND ONLY WITH VALID JSON. No markdown, no backticks."),
        HumanMessage(content=user),
    ])
    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def analyze_medical_report(text: str, doc_type: str, language: str = "English") -> dict:
    """Analyze a medical report and explain it in simple language"""
    logger.info(f"Analyzing {doc_type} in {language}")

    lang_instruction = f"Respond in {language}." if language != "English" else ""

    system = f"""You are a compassionate medical AI assistant that helps patients understand their medical documents.
Your role is to:
1. Explain medical findings in simple, clear language that non-medical people can understand
2. Highlight what is normal vs abnormal
3. Explain what each test/finding means
4. Never diagnose — always recommend consulting a doctor
5. Be reassuring and calm in tone
{lang_instruction}"""

    type_instructions = {
        "lab_report": "Extract each test, its value, normal range, and explain what it means in simple terms.",
        "prescription": "List each medication, its purpose, dosage instructions, and common side effects to watch for.",
        "radiology_report": "Explain the imaging findings in simple terms, highlight key findings.",
        "discharge_summary": "Summarize the hospital visit, diagnosis, treatment received, and follow-up instructions.",
        "pathology_report": "Explain the pathology findings in simple terms.",
        "general_medical": "Analyze and explain the medical document in simple language.",
    }

    prompt = f"""{type_instructions.get(doc_type, type_instructions['general_medical'])}

MEDICAL DOCUMENT:
{text[:4000]}

Provide:
1. **Summary** — What this document shows in 2-3 simple sentences
2. **Key Findings** — List the most important findings with simple explanations
3. **Normal vs Abnormal** — What's within normal range and what needs attention
4. **What This Means For You** — Practical, reassuring explanation
5. **Questions to Ask Your Doctor** — 3-5 specific questions based on these results

Write in simple language. Avoid jargon. Be compassionate and clear."""

    try:
        analysis = call_llm(system, prompt, temperature=0.1)
        return {
            "analysis": analysis + DISCLAIMER,
            "doc_type": doc_type,
            "language": language,
            "success": True,
        }
    except Exception as e:
        return {"analysis": f"Analysis error: {str(e)}", "success": False}


def summarize_prescription(text: str, language: str = "English") -> dict:
    """Extract and explain prescription information"""
    lang_instruction = f"Respond in {language}." if language != "English" else ""

    system = f"""You are a helpful pharmacist assistant explaining prescriptions to patients.
Be clear, safe, and always recommend following doctor's instructions exactly.
{lang_instruction}"""

    prompt = f"""Analyze this prescription and provide a patient-friendly summary:

{text[:3000]}

Return a structured explanation with:
1. **Medications List** — For each medication:
   - Name (generic + brand if mentioned)
   - What it's used for (purpose)
   - How to take it (dosage & timing)
   - Important warnings or side effects
   - Food/drug interactions if mentioned
2. **Important Instructions** — Key things to remember
3. **When to Call Your Doctor** — Warning signs to watch for
4. **Refill Information** — Number of refills if mentioned

Write in simple, clear language."""

    try:
        summary = call_llm(system, prompt, temperature=0.1)
        return {
            "summary": summary + DISCLAIMER,
            "success": True,
            "language": language,
        }
    except Exception as e:
        return {"summary": f"Error: {str(e)}", "success": False}


def extract_lab_values(text: str) -> dict:
    """Extract structured lab values from text"""
    try:
        result = call_llm_json(
            system="You are a medical data extraction specialist. Extract lab values accurately.",
            user=f"""Extract all lab test values from this report:

{text[:3000]}

Return JSON:
{{
  "lab_values": [
    {{
      "test_name": "Test name",
      "value": "measured value with unit",
      "normal_range": "normal range with unit",
      "status": "normal|low|high|critical",
      "interpretation": "1 sentence plain English explanation"
    }}
  ],
  "collection_date": "date if mentioned or null",
  "patient_info": "patient name/ID if visible or null",
  "ordering_doctor": "doctor name if mentioned or null"
}}"""
        )
        return result
    except Exception as e:
        return {"lab_values": [], "error": str(e)}


def answer_medical_question(
    question: str,
    context: str = "",
    language: str = "English",
    conversation_history: List[dict] = None,
) -> str:
    """Answer patient medical questions with context from their documents"""
    lang_instruction = f"Respond in {language}." if language != "English" else ""

    system = f"""You are a compassionate, knowledgeable medical AI assistant.
Your role is to help patients understand their health information.

Rules:
1. Answer clearly in simple language
2. Reference the patient's document context when relevant
3. NEVER diagnose conditions
4. ALWAYS recommend consulting a doctor for medical decisions
5. Be empathetic and reassuring
6. If the question is beyond your scope, say so and recommend a specialist
{lang_instruction}"""

    context_section = f"\nPatient's medical document context:\n{context[:2000]}\n" if context else ""

    # Build conversation history
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    llm = get_llm(temperature=0.2)

    messages = [SystemMessage(content=system)]
    if context_section:
        messages.append(HumanMessage(content=f"Here is my medical document context:{context_section}"))
        messages.append(AIMessage(content="I've reviewed your medical document. Please go ahead and ask your questions."))

    # Add conversation history
    if conversation_history:
        for msg in conversation_history[-6:]:  # last 6 messages
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=question))

    try:
        response = llm.invoke(messages)
        answer = response.content.strip()
        return answer + DISCLAIMER
    except Exception as e:
        return f"I couldn't process your question: {str(e)}" + DISCLAIMER


def translate_medical_text(text: str, target_language: str) -> str:
    """Translate medical text to another language in simple terms"""
    system = f"""You are a medical translator. Translate medical content to {target_language}.
Use simple, everyday words where possible. Keep medical terms but explain them."""

    try:
        return call_llm(system, f"Translate this medical information to {target_language}:\n\n{text[:2000]}", temperature=0.1)
    except Exception as e:
        return f"Translation error: {str(e)}"


def generate_health_summary(analyses: List[dict]) -> str:
    """Generate an overall health summary from multiple analyses"""
    if not analyses:
        return "No analyses available."

    combined = "\n\n".join(a.get("analysis", a.get("summary", "")) for a in analyses[:5])

    system = "You are a medical AI. Create a clear, simple patient health summary."
    prompt = f"""Based on these medical document analyses, create a concise patient health summary:

{combined[:3000]}

Write a 3-4 paragraph summary covering:
1. Overall health picture
2. Key areas of concern (if any)
3. Positive findings
4. Recommended next steps"""

    try:
        return call_llm(system, prompt, temperature=0.2) + DISCLAIMER
    except Exception as e:
        return f"Summary error: {str(e)}"
