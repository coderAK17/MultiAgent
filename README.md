# MultiAgent
MultiAgent Systeam
# 🧠 Smart Document Intelligence System

An AI-powered multi-agent document analysis platform that intelligently routes documents through specialized pipelines — delivering **clinical-grade medical analysis** and **semantic research retrieval** from a single upload.

---

## 🚀 Overview

Most document AI systems treat every file the same. This system doesn't.

A **Classification Agent** acts as the brain — it detects the document type first, then routes it through the optimal pipeline. Medical documents get full-context LLM analysis (zero chunking, zero retrieval loss). Everything else goes through a high-performance RAG pipeline.

---

## ⚙️ Architecture

```
                        USER UPLOAD
                             │
                             ▼
               ┌─────────────────────────┐
               │   Classification Agent   │
               └─────────────────────────┘
                      /             \
                     /               \
                    ▼                 ▼
          MEDICAL PIPELINE        RAG PIPELINE
                 │                      │
                 ▼                      ▼
       Full LLM Context Analysis   Chunk → Embed → Retrieve
                 │                      │
                 ▼                      ▼
      ┌─────────────────┐      ┌─────────────────┐
      │ Diagnosis Agent  │      │ Research Agent   │
      │ Triage Agent     │      │ Summarizer Agent │
      │ Summary Agent    │      └─────────────────┘
      └─────────────────┘
```

---

## 🧬 Medical Pipeline

Handles: Blood reports, MRI scans, Prescriptions, Lab reports, Discharge summaries

**Why no RAG for medical?**
> Medical reports are concise but highly context-sensitive. To avoid retrieval loss and preserve diagnostic integrity, we process the full report directly through the LLM instead of chunk-based retrieval.

**Flow:**
```
Upload PDF → OCR/Text Extraction → Full LLM Analysis
→ Extract Symptoms, Diseases, Abnormal Values, Severity, Risk
→ Triage Classification → Doctor-Ready Summary
```

---

## 📚 RAG Pipeline

Handles: Research papers, Large notes, Policy documents, Articles, Academic PDFs

**Flow:**
```
Upload PDF → Chunking → Embeddings → Vector DB
→ Semantic Retrieval → Research Agent → Summary
```

---

## 🗂️ Project Structure

```
backend/
│
├── agents/
│   ├── classifier_agent.py      # Document type detection
│   ├── diagnosis_agent.py       # Medical diagnosis extraction
│   ├── triage_agent.py          # Severity & urgency classification
│   ├── summary_agent.py         # Report summarization
│   ├── research_agent.py        # RAG-based research Q&A
│   └── orchestrator.py          # Master pipeline controller
│
├── pipelines/
│   ├── medical_pipeline.py      # Full-context medical flow
│   └── rag_pipeline.py          # Chunk-embed-retrieve flow
│
├── rag/
│   ├── chunker.py
│   ├── embeddings.py
│   ├── retriever.py
│   └── vectorstore.py
│
├── medical/
│   ├── report_parser.py
│   ├── symptom_extractor.py
│   ├── abnormality_detector.py
│   └── medical_prompting.py
│
├── uploads/
└── api/
```

---

## 🤖 Classification Agent

Detects document category using a combination of:
- Filename analysis
- Medical keyword matching (`Hemoglobin`, `MRI`, `CBC`, `BP`, `Prescription`, etc.)
- LLM-based classification fallback

**Supported Categories:**
| Category | Example Documents |
|---|---|
| 🏥 Medical | Blood reports, MRI, Lab results |
| 🔬 Research | Papers, Journals, Studies |
| ⚖️ Legal | Contracts, Policies |
| 💰 Financial | Invoices, Reports |
| 🎓 Academic | Notes, Textbooks |
| 📄 General | Any other PDF |

---

## 🛠️ Tech Stack

- **LLM** — Claude / GPT (via API)
- **OCR** — PyMuPDF / Tesseract
- **Embeddings** — OpenAI / Sentence Transformers
- **Vector DB** — FAISS / ChromaDB / Pinecone
- **Backend** — Python (FastAPI)
- **Orchestration** — Custom multi-agent framework

---

## 📦 Installation

```bash
git clone https://github.com/your-username/smart-doc-intelligence
cd smart-doc-intelligence
pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
uvicorn api.main:app --reload
```

---

## 🏆 Key Differentiators

- **Dual-pipeline architecture** — not one-size-fits-all
- **Medical-safe processing** — no chunking on sensitive reports
- **Multi-agent orchestration** — specialized agents for each task
- **Scalable RAG** — handles large documents efficiently

---

## 👨‍💻 Author

Built with 🔥 by [Your Name]  
AI & Data Science Engineer

