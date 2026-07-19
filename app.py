"""
Medsplain — Streamlit UI
Medical Report Analysis · OCR · Prescription Summary · Patient Q&A · Multi-language
"""

import os
import tempfile
from pathlib import Path
import streamlit as st
from src.tools.medical_analysis import SUPPORTED_LANGUAGES, answer_medical_question

st.set_page_config(
    page_title="Medsplain",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main, .stApp { background-color: #0f1117; }
.hero { font-size:2.4rem; font-weight:800;
  background:linear-gradient(135deg,#00897b,#1565c0);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.sub { color:#888; font-size:1rem; margin-bottom:1rem; }
.card { background:#1e2130; border:1px solid #2d3148; border-radius:12px; padding:20px 24px; margin:10px 0; }
.disclaimer { background:#1a1a2e; border:2px solid #f9a825; border-radius:10px; padding:14px 18px; margin:12px 0; }
.chat-user { background:#1a1d2e; border-left:3px solid #1565c0; border-radius:8px; padding:12px 16px; margin:8px 0; }
.chat-ai { background:#1a2e1a; border-left:3px solid #00897b; border-radius:8px; padding:12px 16px; margin:8px 0; }
.lab-normal { color:#43b89c; font-weight:600; }
.lab-high { color:#ef5350; font-weight:600; }
.lab-low { color:#f9a825; font-weight:600; }
.lab-critical { color:#ef5350; font-weight:800; }
div[data-testid="stSidebar"] { background:#151825; }
.stButton>button { background:linear-gradient(135deg,#00897b,#1565c0);
  color:#fff; border:none; border-radius:8px; font-weight:700; width:100%; }
.stTextInput input, .stTextArea textarea { background:#1e2130 !important;
  color:#e0e0e0 !important; border:1px solid #2d3148 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💊 Medsplain")
    st.markdown("---")

    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...",
                              help="Get free key at console.groq.com")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    anthropic_key = st.text_input("Anthropic API Key (optional)", type="password", placeholder="sk-ant-...",
                                   help="Uses Claude for better medical analysis")
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key

    st.markdown("---")
    language = st.selectbox("🌍 Language", list(SUPPORTED_LANGUAGES.keys()))

    st.markdown("---")
    st.markdown("### 🔗 Features")
    for f in ["📄 Medical Report Analysis", "🔬 Lab Value Extraction", "💊 Prescription Summary", "📸 OCR Document Reading", "💬 Patient Q&A", "🌍 Multi-language"]:
        st.markdown(f)

    st.markdown("---")
    st.markdown('<div style="background:#1a1a2e;border:1px solid #f9a825;border-radius:8px;padding:10px;font-size:.8rem;color:#f9a825;">⚠️ For informational purposes only. Always consult a qualified healthcare professional.</div>', unsafe_allow_html=True)

    if st.button("🗑️ Clear Session"):
        for key in ["healthcare_state", "chat_history", "last_context"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero">⚕️ Medsplain</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Upload medical reports, lab results, or prescriptions — get simple, clear explanations in your language</div>', unsafe_allow_html=True)

st.markdown("""<div class="disclaimer">
⚠️ <b>Medical Disclaimer:</b> This AI assistant is for informational purposes only and does not constitute medical advice, diagnosis, or treatment.
Always consult a qualified healthcare professional for medical decisions.
</div>""", unsafe_allow_html=True)

# ── Init session state ─────────────────────────────────────────────────────────
if "healthcare_state" not in st.session_state:
    st.session_state.healthcare_state = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_context" not in st.session_state:
    st.session_state.last_context = ""

# ── Main tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📄 Analyze Document", "🔬 Lab Values", "💬 Ask Questions", "💊 Prescription"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1: Document Analysis
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📄 Upload Medical Document")
    st.caption("Supports: PDF, PNG, JPG, DOCX, TXT — up to 50 pages")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your medical document",
            type=["pdf", "png", "jpg", "jpeg", "docx", "txt"],
        )
    with col2:
        st.markdown("#### Or paste text directly")
        pasted_text = st.text_area("Paste medical text", height=120,
                                    placeholder="Paste lab report, prescription, or any medical text...")

    analyze_btn = st.button("🔬 Analyze Document", use_container_width=True)

    if analyze_btn:
        if not os.getenv("GROQ_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            st.error("⚠️ Add your API key in the sidebar.")
            st.stop()

        if not uploaded_file and not pasted_text.strip():
            st.error("Please upload a file or paste text.")
            st.stop()

        with st.spinner("🔬 Analyzing your medical document..."):
            try:
                from src.tools.ocr_tool import detect_document_type
                from src.tools.medical_analysis import analyze_medical_report, summarize_prescription, extract_lab_values

                if pasted_text.strip():
                    text = pasted_text
                    method = "pasted_text"
                else:
                    # Save uploaded file
                    suffix = Path(uploaded_file.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    from src.tools.ocr_tool import extract_text_from_file
                    ocr_result = extract_text_from_file(tmp_path)
                    text = ocr_result["text"]
                    method = ocr_result["method"]

                    st.info(f"✅ Text extracted via {method} — {len(text.split())} words")

                # Detect and analyze
                doc_type = detect_document_type(text)
                st.session_state.last_context = text

                st.markdown(f"**📋 Document Type Detected:** {doc_type.replace('_', ' ').title()}")

                if doc_type == "prescription":
                    result = summarize_prescription(text, language)
                    summary_text = result.get("summary", "")
                    st.markdown("### 💊 Prescription Summary")
                    st.markdown(summary_text)
                    st.session_state.healthcare_state = {"type": "prescription", "result": result, "text": text}
                else:
                    result = analyze_medical_report(text, doc_type, language)
                    analysis_text = result.get("analysis", "")
                    st.markdown(f"### 🔬 Analysis — {doc_type.replace('_', ' ').title()}")
                    st.markdown(analysis_text)

                    if doc_type == "lab_report":
                        lab_result = extract_lab_values(text)
                        st.session_state.healthcare_state = {
                            "type": doc_type, "result": result,
                            "lab_values": lab_result, "text": text
                        }
                    else:
                        st.session_state.healthcare_state = {"type": doc_type, "result": result, "text": text}

                st.success("✅ Analysis complete! Ask questions in the 'Ask Questions' tab.")

            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
                st.exception(e)

    # Landing
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""<div class="card"><b>🔬 Lab Reports</b><br>
            <span style="color:#888">Blood tests, urine analysis, cholesterol, thyroid, liver function — explained simply</span></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""<div class="card"><b>📋 Medical Reports</b><br>
            <span style="color:#888">Radiology, discharge summaries, pathology — what do the findings mean?</span></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown("""<div class="card"><b>💊 Prescriptions</b><br>
            <span style="color:#888">What each medication does, how to take it, side effects to watch for</span></div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2: Lab Values
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🔬 Lab Value Breakdown")

    state = st.session_state.healthcare_state
    if state and state.get("lab_values"):
        lab_data = state["lab_values"]
        values = lab_data.get("lab_values", [])

        if lab_data.get("collection_date"):
            st.caption(f"Collection date: {lab_data['collection_date']}")

        if values:
            # Summary metrics
            normal_count = sum(1 for v in values if v.get("status") == "normal")
            abnormal_count = len(values) - normal_count
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Tests", len(values))
            c2.metric("Normal", normal_count)
            c3.metric("Needs Attention", abnormal_count)

            st.markdown("---")
            for val in values:
                status = val.get("status", "normal")
                status_colors = {"normal": "🟢", "high": "🔴", "low": "🟡", "critical": "🚨"}
                icon = status_colors.get(status, "⚪")

                with st.expander(f"{icon} {val.get('test_name', 'Test')} — {val.get('value', 'N/A')}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Value:** {val.get('value', 'N/A')}")
                        st.markdown(f"**Normal Range:** {val.get('normal_range', 'N/A')}")
                    with col2:
                        status_cls = {"normal": "lab-normal", "high": "lab-high", "low": "lab-low", "critical": "lab-critical"}.get(status, "lab-normal")
                        st.markdown(f'**Status:** <span class="{status_cls}">{status.upper()}</span>', unsafe_allow_html=True)
                    st.markdown(f"**What this means:** {val.get('interpretation', '')}")
        else:
            st.info("No structured lab values could be extracted. View the full analysis in the Analysis tab.")
    else:
        st.info("Upload a lab report in the 'Analyze Document' tab first.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3: Patient Q&A
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 💬 Ask Questions About Your Health")

    # Example questions
    if st.session_state.healthcare_state:
        examples = [
            "What does this report mean for my health?",
            "Which values are abnormal and should I be worried?",
            "What lifestyle changes should I consider?",
            "What questions should I ask my doctor?",
            "Can you explain this in simpler terms?",
        ]
        st.markdown("**💡 Example questions:**")
        ex_cols = st.columns(3)
        for i, ex in enumerate(examples[:3]):
            with ex_cols[i]:
                if st.button(ex, key=f"qex{i}"):
                    st.session_state["pending_q"] = ex

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">👤 <b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-ai">🤖 <b>Healthcare AI:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)

    # Question input
    st.markdown("---")
    question = st.text_input(
        "Ask a question",
        value=st.session_state.pop("pending_q", ""),
        placeholder="e.g. What does high creatinine mean? Is my blood sugar normal?",
    )

    if st.button("💬 Ask", use_container_width=True):
        if not question.strip():
            st.error("Please enter a question.")
            st.stop()
        if not os.getenv("GROQ_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            st.error("⚠️ Add your API key in the sidebar.")
            st.stop()

        with st.spinner("🤔 Thinking..."):
            answer = answer_medical_question(
                question=question,
                context=st.session_state.last_context,
                language=language,
                conversation_history=st.session_state.chat_history,
            )
            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

    if not st.session_state.healthcare_state and not st.session_state.chat_history:
        st.info("You can ask general health questions without uploading a document, or upload one first for document-specific answers.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4: Prescription
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 💊 Prescription Summary")
    st.caption("Upload a prescription or paste it below to get a clear explanation of your medications")

    rx_text = st.text_area("Paste prescription text", height=200,
                            placeholder="""Example:
Rx: Metformin 500mg
Sig: Take 1 tablet twice daily with meals
Disp: 60 tablets
Refills: 3

Rx: Lisinopril 10mg
Sig: Take 1 tablet once daily
Disp: 30 tablets
Refills: 6""")

    if st.button("💊 Summarize Prescription", use_container_width=True):
        if not rx_text.strip():
            st.error("Please paste a prescription.")
            st.stop()
        if not os.getenv("GROQ_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            st.error("⚠️ Add your API key in the sidebar.")
            st.stop()

        with st.spinner("Analyzing prescription..."):
            from src.tools.medical_analysis import summarize_prescription
            result = summarize_prescription(rx_text, language)
            if result["success"]:
                st.markdown("### Your Medication Guide")
                st.markdown(result["summary"])
            else:
                st.error(f"Error: {result.get('summary', 'Unknown error')}")
