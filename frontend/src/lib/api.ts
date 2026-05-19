import axios from "axios";

const DEFAULT_API = "http://localhost:8000";
export const API_BASE =
  (typeof import.meta !== "undefined" && (import.meta as any).env?.VITE_API_BASE) || DEFAULT_API;

export const http = axios.create({
  baseURL: API_BASE,
  timeout: 600000, // 10 minutes to allow for large files and first-time model downloads
});

export type DocCategory = "medical" | "research" | "legal" | "financial" | "academic" | "general" | "task" | "startup";

export interface ClassificationResult { category: DocCategory; confidence?: number; reasoning?: string; }
export interface MedicalFinding {
  symptoms?: string[]; diseases?: string[];
  abnormal_values?: { name: string; value: string; reference?: string; severity?: string }[];
  severity?: "low" | "moderate" | "high" | "critical"; triage?: string; summary?: string;
}
export interface RagResult { summary?: string; key_points?: string[]; citations?: { chunk: string; score?: number }[]; }

export interface TaskResult { plan?: string; execution?: string; review?: string; }
export interface StartupSimulation { vision?: string; architecture?: string; scoping?: string; }
export interface AutomatedResearch { search_queries?: string[]; summary?: string; presentation?: string; }

export interface AnalyzeResponse {
  classification: ClassificationResult;
  pipeline: "medical" | "rag" | "task" | "startup" | "research";
  medical?: MedicalFinding; 
  research?: RagResult; 
  task_team?: TaskResult;
  startup?: StartupSimulation;
  automated_research?: AutomatedResearch;
  raw_text_preview?: string;
}

export async function uploadDocument(file: File): Promise<AnalyzeResponse> {
  const fd = new FormData();
  fd.append("file", file);
  const { data } = await http.post<AnalyzeResponse>("/upload", fd, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function chat(message: string, history?: any[], context?: string, sessionId?: string) {
  const { data } = await http.post("/chat", { message, session_id: sessionId, history, context });
  return data as { reply: string; agents?: string[] };
}

export async function getAgentStatus() {
  const { data } = await http.get("/agents/status");
  return data;
}

export async function getDocuments() {
  const { data } = await http.get("/documents");
  return data;
}

export async function getAnalysis(id: string) {
  const { data } = await http.get(`/analysis/${id}`);
  return data;
}
