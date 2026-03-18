import { Highlight } from '../api/client';

export const highlightCode = (code: string, highlights: Highlight[] = []): { text: string; spans: Array<{ start: number; end: number; type: string }> } => {
  return {
    text: code,
    spans: highlights,
  };
};

export const getHighlightColor = (type: string): string => {
  switch (type) {
    case 'plagiarism':
      return '#fca5a5'; // light red
    case 'ai-detected':
      return '#fbbf24'; // light amber
    default:
      return 'transparent';
  }
};

export const calculateReadingTime = (code: string): number => {
  const wordsPerMinute = 200;
  const words = code.trim().split(/\s+/).length;
  return Math.ceil(words / wordsPerMinute);
};
