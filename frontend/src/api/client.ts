/**
 * API client — all backend calls go through here.
 *
 * Base URL defaults to http://localhost:5000 for development.
 * Override with VITE_API_URL env var in .env or at build time.
 */

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:5000';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(body.error || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

/* ---------- Types ---------- */

export interface UploadResult {
  file_id: string;
  filename: string;
  stored_name: string;
}

export interface RunStartResult {
  run_id: string;
  status: string;
}

export interface RunSummary {
  run_id: string;
  timestamp: string;
  input_filename: string;
  status: string;
  issues: string[];
  genai_model: string;
}

export interface RunDetail {
  run_id: string;
  timestamp: string;
  input_filename: string;
  status: string;
  metadata_summary: Record<string, string>;
  issues: string[];
  metrics_before: Record<string, number>;
  metrics_after: Record<string, number>;
  plan_json: string;
  validation: Record<string, unknown>;
  applied_ops: string[];
  explainability: Record<string, unknown>;
  report_path: string;
  before_after_path: string;
  agent_logs: LogEntry[];
  genai_model: string;
  genai_llm_calls: number;
  chat_history: ChatMessage[];
}

export interface LogEntry {
  timestamp?: string;
  phase?: string;
  event?: string;
  detail?: string;
}

export interface ChatMessage {
  role: string;
  content: string;
  timestamp?: string;
}

export interface RunStatusResult {
  run_id: string;
  status: string;
}

/* ---------- API Functions ---------- */

export async function uploadFile(file: File): Promise<UploadResult> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API_BASE}/api/upload`, { method: 'POST', body: form });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(body.error || `Upload failed: ${res.status}`);
  }
  return res.json();
}

export async function startRun(params: {
  file_id: string;
  genai?: boolean;
  model?: string;
  max_iters?: number;
}): Promise<RunStartResult> {
  return request('/api/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
}

export async function pollStatus(runId: string): Promise<RunStatusResult> {
  return request(`/api/runs/${runId}/status`);
}

export async function getRuns(): Promise<RunSummary[]> {
  const data = await request<{ runs: RunSummary[] }>('/api/runs');
  return data.runs;
}

export async function getRun(runId: string): Promise<RunDetail> {
  return request(`/api/runs/${runId}`);
}

export async function getReport(runId: string): Promise<string> {
  const data = await request<{ markdown: string }>(`/api/runs/${runId}/report`);
  return data.markdown;
}

export function getBeforeAfterUrl(runId: string): string {
  return `${API_BASE}/api/runs/${runId}/before_after`;
}

export async function sendChat(
  runId: string,
  message: string,
  model?: string,
): Promise<string> {
  const data = await request<{ reply: string }>(`/api/runs/${runId}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, model }),
  });
  return data.reply;
}

export async function getLogs(runId: string): Promise<LogEntry[]> {
  const data = await request<{ logs: LogEntry[] }>(`/api/runs/${runId}/logs`);
  return data.logs;
}
