export const SUPPORTED_LANGUAGES = [
  'python',
  'javascript',
  'typescript',
  'java',
  'c',
  'cpp',
  'go',
  'rust',
];

export const CONFIDENCE_COLORS: Record<string, string> = {
  high: '#ef4444',
  medium: '#f59e0b',
  low: '#10b981',
  unknown: '#6b7280',
};

export const getConfidenceColor = (confidence: string): string => {
  return CONFIDENCE_COLORS[confidence.toLowerCase()] || CONFIDENCE_COLORS.unknown;
};

export const getLanguageIcon = (language: string): string => {
  const icons: Record<string, string> = {
    python: '🐍',
    javascript: '📜',
    typescript: '📘',
    java: '☕',
    c: '⚙️',
    cpp: '⚙️',
    go: '🐹',
    rust: '🦀',
  };
  return icons[language.toLowerCase()] || '💻';
};

export const normalizeScoreToPercent = (value: number): number => {
  if (!Number.isFinite(value)) return 0;
  if (value < 0) return 0;
  if (value <= 1) return value * 100;
  if (value > 100) return 100;
  return value;
};

export const formatPercentage = (value: number): string => {
  return normalizeScoreToPercent(value).toFixed(1);
};

export const formatLatency = (ms: number): string => {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
};
