import os
import uuid
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb import EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from schema import ClassificationResult, MedicalFinding, RagResult, Citation, TaskResult, StartupSimulation, AutomatedResearch

load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

# --- Custom embedding function using sentence-transformers (PyTorch, NOT ONNX) ---
# This bypasses ChromaDB's default ONNX download (79MB) and uses the lighter
# PyTorch model (~22MB) cached in ~/.cache/huggingface after the first run.
_embedding_model = None

class SentenceTransformerEF(EmbeddingFunction):
    """Wrap sentence-transformers to provide embeddings for ChromaDB."""
    def __init__(self):
        global _embedding_model
        if _embedding_model is None:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.model = _embedding_model

    def __call__(self, input: List[str]) -> Embeddings:
        return self.model.encode(input, convert_to_numpy=True).tolist()

def classify_document(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Router Prompt: "You are a classifier. Look at this text and output strictly 'medical' or 'general'."
    """
    text = state.get("extracted_text", "")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a classifier. Look at this text and classify it into one of: 'medical', 'research', 'legal', 'financial', 'academic', 'general'."),
        ("human", "Text:\n{text}")
    ])
    
    chain = prompt | llm.with_structured_output(ClassificationResult)
    classification: ClassificationResult = chain.invoke({"text": text[:2000]}) # Limit text length for classification
    
    cat = classification.category
    doc_type = cat if cat in ["medical", "task", "startup", "research"] else "general"
        
    return {"doc_type": doc_type, "classification": classification}

def analyze_medical(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Diagnostician Prompt: "You are an expert medical AI. Analyze the provided symptoms and clinical text, then generate a structured summary."
    """
    text = state.get("extracted_text", "")
    symptoms = state.get("user_symptoms", "")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert medical AI. Analyze the provided symptoms and clinical text, then generate a structured summary including Diagnosis, Triage recommendations, abnormal values, and severity."),
        ("human", "Symptoms: {symptoms}\n\nClinical Text:\n{text}")
    ])
    
    chain = prompt | llm.with_structured_output(MedicalFinding)
    finding: MedicalFinding = chain.invoke({"symptoms": symptoms, "text": text[:4000]})
    
    return {"medical_finding": finding}

def process_general_rag(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    RAG Pipeline: Chunk -> Embed with sentence-transformers -> Store in ChromaDB
    -> Semantic vector search -> LLM answer generation.
    """
    text = state.get("extracted_text", "")
    query = state.get("user_symptoms", "") or "Summarize the document."

    if not text:
        return {"rag_result": RagResult(summary="No text extracted from document.")}

    # 1. Chunk the document
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = [doc.page_content for doc in splitter.create_documents([text])]

    # 2. Create a fresh ephemeral ChromaDB collection with our PyTorch embedding fn
    ef = SentenceTransformerEF()
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name=f"doc_{uuid.uuid4().hex[:8]}",
        embedding_function=ef,
    )

    # 3. Add all chunks (ChromaDB calls ef() to embed them)
    collection.add(
        documents=chunks,
        ids=[str(i) for i in range(len(chunks))],
    )

    # 4. Semantic similarity search
    results = collection.query(query_texts=[query], n_results=min(3, len(chunks)))
    top_chunks = results["documents"][0] if results["documents"] else chunks[:3]
    context = "\n\n".join(top_chunks)

    # 5. LLM answer over retrieved context
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful research assistant. Answer the query based on the provided context. Extract key points and provide a concise summary."),
        ("human", "Query: {query}\n\nContext:\n{context}")
    ])
    chain = prompt | llm.with_structured_output(RagResult)
    rag_result: RagResult = chain.invoke({"query": query, "context": context})

    # 6. Attach source citations
    rag_result.citations = [Citation(chunk=c[:200] + "...") for c in top_chunks]

    return {"rag_result": rag_result, "rag_context": context}

def run_task_team(state: Dict[str, Any]) -> Dict[str, Any]:
    text = state.get("extracted_text", "")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI Task Team. First plan the execution steps, then execute them, and finally review the output based on the provided text."),
        ("human", "Text:\n{text}")
    ])
    chain = prompt | llm.with_structured_output(TaskResult)
    res = chain.invoke({"text": text[:4000]})
    return {"task_result": res}

def run_startup_sim(state: Dict[str, Any]) -> Dict[str, Any]:
    text = state.get("extracted_text", "")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI Startup Simulator (CEO, CTO, PM). Act based on the given idea or text to outline vision, architecture, and scoping."),
        ("human", "Idea:\n{text}")
    ])
    chain = prompt | llm.with_structured_output(StartupSimulation)
    res = chain.invoke({"text": text[:4000]})
    return {"startup_sim": res}

def run_auto_research(state: Dict[str, Any]) -> Dict[str, Any]:
    text = state.get("extracted_text", "")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an Automated Research Agent. Propose search queries, summarize the topic, and present the final report based on the input."),
        ("human", "Topic:\n{text}")
    ])
    chain = prompt | llm.with_structured_output(AutomatedResearch)
    res = chain.invoke({"text": text[:4000]})
    return {"auto_research": res}
