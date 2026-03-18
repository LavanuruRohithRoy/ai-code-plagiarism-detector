import { useState, useCallback } from 'react';
import { AnalysisResult, analyzeAPI } from '../api/client';
import axios from 'axios';

interface UseAnalysisState {
  loading: boolean;
  error: string | null;
  results: AnalysisResult[];
  latency: number | null;
}

export const useAnalysis = () => {
  const [state, setState] = useState<UseAnalysisState>({
    loading: false,
    error: null,
    results: [],
    latency: null,
  });

  const getErrorMessage = (err: unknown): string => {
    if (axios.isAxiosError(err)) {
      const detail = err.response?.data?.detail;
      if (typeof detail === 'string' && detail.trim()) {
        return detail;
      }
      return err.message || 'Analysis failed';
    }
    return err instanceof Error ? err.message : 'Analysis failed';
  };

  const analyzeFiles = useCallback(async (files: File[]) => {
    setState({ loading: true, error: null, results: [], latency: null });
    try {
      const data = await analyzeAPI.analyzeFiles(files);
      setState({
        loading: false,
        error: null,
        results: data.results,
        latency: data._latency,
      });
      return data;
    } catch (err) {
      const errorMessage = getErrorMessage(err);
      setState({ loading: false, error: errorMessage, results: [], latency: null });
      throw err;
    }
  }, []);

  const analyzeFile = useCallback(async (file: File) => {
    setState({ loading: true, error: null, results: [], latency: null });
    try {
      const data = await analyzeAPI.analyzeFile(file);
      setState({
        loading: false,
        error: null,
        results: [data as AnalysisResult],
        latency: (data as any)._latency,
      });
      return data;
    } catch (err) {
      const errorMessage = getErrorMessage(err);
      setState({ loading: false, error: errorMessage, results: [], latency: null });
      throw err;
    }
  }, []);

  const reset = useCallback(() => {
    setState({ loading: false, error: null, results: [], latency: null });
  }, []);

  return { ...state, analyzeFiles, analyzeFile, reset };
};
