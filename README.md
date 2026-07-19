# 💊 Medsplain — AI Healthcare Assistant

An AI assistant that explains complex medical reports in simple language, summarizes prescriptions, reads lab reports via OCR, and helps patients better understand their health data.

> ⚠️ Medical Disclaimer: For informational purposes only. Always consult a qualified healthcare professional.

## 🎯 Features

- Medical Report Analysis — explains findings in simple language
- Lab Value Extraction — structured breakdown vs normal ranges
- Prescription Summary — medications, dosage, side effects
- OCR Document Reading — reads scanned PDFs and images
- Patient Q&A — ask follow-up questions about your documents
- Multi-language — 8 languages including Hindi, Spanish, Arabic

## 🏗️ Tech Stack

- LangGraph — Agent pipeline orchestration
- Llama 3.3 70B via Groq — Medical analysis and Q&A
- Claude (optional) — Enhanced medical understanding
- pytesseract + pypdf — OCR document reading
- Streamlit — Interactive UI

## 🚀 Quick Start

pip install -r requirements.txt
export GROQ_API_KEY=gsk_your_key
streamlit run app.py

## 🔑 API Keys

- Groq (required) — console.groq.com — free
- Anthropic (optional) — console.anthropic.com — for Claude

## 🌍 Supported Languages

English, Hindi, Spanish, French, Arabic, Chinese, Portuguese, German

## 🧪 Tests

pytest tests/ -v

## ☁️ Deploy

Streamlit Cloud -> app.py -> add GROQ_API_KEY secret

---
Built with LangGraph + Llama 3 + OCR + Streamlit
