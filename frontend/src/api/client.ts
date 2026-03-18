import axios from 'axios';

export interface Highlight {
  start: number;
  end: number;
  type: 'plagiarism' | 'ai-detected';
}

export interface ExplanationData {
  token_similarity?: number;
  semantic_similarity?: number;
  structure_similarity?: number;
  code_lines?: number;
  size_penalty_applied?: boolean;
  db_inserted?: boolean;
  language?: string;
  metrics?: Record<string, number>;
  signal_bands?: Record<string, string>;
  highlights?: Highlight[];
  source_code?: string;
  highlight_legend?: Record<string, string>;
  known_match?: Record<string, string>;
  reasoning: string;
}

export interface AnalysisResult {
  filename: string;
  language: string;
  plagiarism_percentage: number;
  ai_probability: number;
  confidence: string;
  explanation: ExplanationData;
}

export interface BatchAnalysisResponse {
  total_files: number;
  succeeded: number;
  failed: number;
  results: AnalysisResult[];
  errors: Record<string, string>;
}

const rawApiBaseUrl = import.meta.env.VITE_API_BASE_URL || '/api';
const API_BASE_URL = rawApiBaseUrl.replace('://localhost', '://127.0.0.1');

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

// Track request latency
export function measureLatency() {
  const startTime = performance.now();
  return () => performance.now() - startTime;
}

export const analyzeAPI = {
  // Analyze raw code text
  analyzeCode: async (code: string, language?: string) => {
    const start = performance.now();
    const response = await apiClient.post('/analyze', {
      code,
      language: language || undefined,
    });
    const latency = performance.now() - start;
    return { ...response.data, _latency: latency };
  },

  // Analyze single file
  analyzeFile: async (file: File) => {
    const start = performance.now();
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post<AnalysisResult>(
      '/analyze/file',
      formData,
      {
        headers: { 'Content-Type': 'multipart/form-data' },
      }
    );
    const latency = performance.now() - start;
    return { ...response.data, _latency: latency };
  },

  // Analyze multiple files
  analyzeFiles: async (files: File[]) => {
    const start = performance.now();
    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));

    const response = await apiClient.post<BatchAnalysisResponse>(
      '/analyze/files',
      formData,
      {
        headers: { 'Content-Type': 'multipart/form-data' },
      }
    );
    const latency = performance.now() - start;
    return { ...response.data, _latency: latency };
  },

  // Health check
  health: async () => {
    const response = await apiClient.get('/health');
    return response.data;
  },
};

export default apiClient;
