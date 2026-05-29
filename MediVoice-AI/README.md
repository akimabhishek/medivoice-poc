---
title: MediVoice AI
emoji: 📚
colorFrom: purple
colorTo: red
sdk: docker
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# MediVoice AI — Smart Medical Policy RAG & Voice Assistant

MediVoice AI is an intelligent, voice-enabled Proof of Concept (PoC) designed to streamline medical insurance policy querying and lead verification. Built on a robust Retrieval-Augmented Generation (RAG) architecture, the system allows users to interact with complex insurance documents naturally via text or voice, returning instant, accurate insights, processing latency metrics, and real-time API cost computations.

---

## 🚀 Key Features

* **Omnichannel Interface:** Supports both text queries and real-time microphone audio processing.
* **Intelligent RAG Pipeline:** Automatically ingests, parses, and vectorizes insurance policy PDFs to answer highly specific coverage questions.
* **Advanced Voice Pipeline:** Integrates Automatic Speech Recognition (ASR) to clean and process medical terminology, and speech synthesis (TTS) to read answers aloud.
* **Production-Grade Guardrails:** Real-time rate limiting per client IP to manage traffic and protect underlying APIs.
* **Live Telemetry Dashboard:** Tracks and displays token usage, exact API transaction costs, vector database confidence scores, and round-trip latency for every query.

---

## 🛠️ Tech Stack & Architecture

* **Backend Framework:** FastAPI (Asynchronous Python implementation)
* **Orchestration & Serving:** Uvicorn
* **LLM Engine:** Gemini API via the Google Cloud Generative Language API
* **Rate Limiting:** SlowAPI (Token-bucket rate limiting based on client IP addresses)
* **Data Ingestion:** Automated chunking and vector search for PDF/TXT medical policy sheets
* **UI Engine:** Jinja2 HTML Templates & TailwindCSS (served statically via FastAPI)

---

## 📦 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/medivoice-poc.git](https://github.com/your-username/medivoice-poc.git)
cd medivoice-poc

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

pip install -r requirements.txt

GEMINI_API_KEY=your_actual_restricted_gcp_api_key

uvicorn app.main:app --reload

# git add .
# git commit -m "Describe what you changed"
# For main github: git push origin main
# For huggingface: git subtree push --prefix MediVoice-AI huggingface main
