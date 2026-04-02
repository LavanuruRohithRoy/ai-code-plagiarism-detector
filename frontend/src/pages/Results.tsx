import React, { useState, memo } from 'react';
import './Results.css';
import { AnalysisResult } from '../api/client';
import CodeDisplay from '../components/CodeDisplay';
import MetricsPanel from '../components/MetricsPanel';
import { getLanguageIcon, formatLatency, normalizeScoreToPercent } from '../utils/constants';

interface ResultsPageProps {
  data: any;
  onBack: () => void;
}

type ResultTab = 'code' | 'metrics';

const ResultsPage = memo<ResultsPageProps>(({ data, onBack }) => {
  const [selectedResultIndex, setSelectedResultIndex] = useState(0);
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<ResultTab>('code');

  // Handle both single and batch results
  const results: AnalysisResult[] = data.single
    ? [data.single]
    : data.batch?.results || [];

  const latency = data.latency;
  const selectedResult = results[selectedResultIndex];

  const isBatch = results.length > 1;
  const totalPlagiarism = results.length
    ? results.reduce(
        (sum, r) => sum + normalizeScoreToPercent(r.plagiarism_percentage),
        0
      ) / results.length
    : 0;
  const totalAI = results.length
    ? results.reduce((sum, r) => sum + normalizeScoreToPercent(r.ai_probability), 0) /
      results.length
    : 0;

  const triggerDownload = (filename: string, content: string, mimeType: string) => {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const handleExportJson = () => {
    const payload = {
      generated_at: new Date().toISOString(),
      total_files: results.length,
      latency,
      results,
    };
    triggerDownload('analysis-report.json', JSON.stringify(payload, null, 2), 'application/json');
  };

  const escapeCsv = (value: string | number) => {
    const text = String(value ?? '');
    if (text.includes(',') || text.includes('"') || text.includes('\n')) {
      return `"${text.replace(/"/g, '""')}"`;
    }
    return text;
  };

  const handleExportCsv = () => {
    const headers = [
      'filename',
      'language',
      'plagiarism_percentage',
      'ai_probability',
      'confidence',
      'token_similarity',
      'semantic_similarity',
      'structure_similarity',
      'code_lines',
    ];

    const rows = results.map((result) => [
      result.filename,
      result.language,
      normalizeScoreToPercent(result.plagiarism_percentage).toFixed(2),
      normalizeScoreToPercent(result.ai_probability).toFixed(2),
      result.confidence,
      result.explanation.token_similarity ?? '',
      result.explanation.semantic_similarity ?? '',
      result.explanation.structure_similarity ?? '',
      result.explanation.code_lines ?? '',
    ]);

    const csv = [headers, ...rows]
      .map((row) => row.map((cell) => escapeCsv(cell as string | number)).join(','))
      .join('\n');

    triggerDownload('analysis-report.csv', csv, 'text/csv;charset=utf-8;');
  };

  const handleExportPdf = () => {
    if (!selectedResult) {
      return;
    }

    const printWindow = window.open('', '_blank');
    if (!printWindow) {
      return;
    }

    const codeText = selectedResult.explanation.source_code || '';
    const escapedCode = codeText
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    printWindow.document.write(`
      <html>
        <head>
          <title>Analysis Report</title>
          <style>
            body { font-family: Arial, sans-serif; padding: 24px; }
            h1, h2 { margin-bottom: 8px; }
            .meta { margin-bottom: 16px; color: #444; }
            pre { background: #f6f8fa; padding: 12px; border-radius: 6px; overflow: auto; }
          </style>
        </head>
        <body>
          <h1>AI Code Plagiarism Detector - Report</h1>
          <div class="meta">
            <div><strong>File:</strong> ${selectedResult.filename}</div>
            <div><strong>Language:</strong> ${selectedResult.language}</div>
            <div><strong>Plagiarism:</strong> ${normalizeScoreToPercent(selectedResult.plagiarism_percentage).toFixed(1)}%</div>
            <div><strong>AI Probability:</strong> ${normalizeScoreToPercent(selectedResult.ai_probability).toFixed(1)}%</div>
            <div><strong>Confidence:</strong> ${selectedResult.confidence}</div>
          </div>
          <h2>Reasoning</h2>
          <p>${selectedResult.explanation.reasoning}</p>
          <h2>Source Code</h2>
          <pre>${escapedCode}</pre>
        </body>
      </html>
    `);
    printWindow.document.close();
    printWindow.focus();
    printWindow.print();
  };

  return (
    <div className="results-page">
      <div className="results-container">
        {/* Header */}
        <div className="results-header">
          <button className="back-btn" onClick={onBack}>
            ← Back to Upload
          </button>
          <div className="header-content">
            <h1 className="title">Analysis Results</h1>
            {isBatch && (
              <p className="subtitle">
                {results.length} files analyzed • Average Plagiarism:{' '}
                {totalPlagiarism.toFixed(1)}%
              </p>
            )}
          </div>
          {latency && (
            <div className="latency-badge">
              ⚡ {formatLatency(latency)}
            </div>
          )}
        </div>

        {/* Batch Summary */}
        {isBatch && (
          <div className="batch-summary">
            <div className="summary-card">
              <h3>📊 Summary</h3>
              <div className="summary-grid">
                <div className="summary-item">
                  <label>Total Files</label>
                  <span>{results.length}</span>
                </div>
                <div className="summary-item">
                  <label>Avg. Plagiarism</label>
                  <span>{totalPlagiarism.toFixed(1)}%</span>
                </div>
                <div className="summary-item">
                  <label>Avg. AI Probability</label>
                  <span>{totalAI.toFixed(1)}%</span>
                </div>
                <div className="summary-item">
                  <label>Suspicious Files</label>
                  <span>
                    {results.filter((r) => normalizeScoreToPercent(r.plagiarism_percentage) > 30)
                      .length}
                  </span>
                </div>
              </div>
            </div>

            {/* Results List */}
            <div className="results-list">
              <h3>📁 File Results</h3>
              {results.map((result, idx) => (
                <div
                  key={idx}
                  className={`result-item ${
                    selectedResultIndex === idx ? 'selected' : ''
                  } ${expandedIndex === idx ? 'expanded' : ''}`}
                  onClick={() => setSelectedResultIndex(idx)}
                >
                  <div className="result-item-header">
                    <div className="result-icon">
                      {getLanguageIcon(result.language || 'unknown')}
                    </div>
                    <div className="result-name">
                      <div className="file-name">{result.filename}</div>
                      <div className="file-language">{result.language}</div>
                    </div>
                    <div className="result-scores">
                      <div
                        className="plagiarism-badge"
                        style={{
                          color:
                            normalizeScoreToPercent(result.plagiarism_percentage) > 70
                              ? '#991b1b'
                              : normalizeScoreToPercent(result.plagiarism_percentage) > 30
                              ? '#b45309'
                              : '#065f46',
                        }}
                      >
                        {normalizeScoreToPercent(result.plagiarism_percentage).toFixed(0)}%
                      </div>
                      <div className="ai-badge">
                        {normalizeScoreToPercent(result.ai_probability).toFixed(0)}%
                      </div>
                    </div>
                    <button
                      className="expand-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        setExpandedIndex(
                          expandedIndex === idx ? null : idx
                        );
                      }}
                    >
                      {expandedIndex === idx ? '▼' : '▶'}
                    </button>
                  </div>
                  {expandedIndex === idx && (
                    <div className="result-details">
                      <div className="detail-row">
                        <span>Confidence:</span>
                        <strong>{result.confidence}</strong>
                      </div>
                      <div className="detail-row">
                        <span>Lines:</span>
                        <strong>
                          {result.explanation.code_lines || 'N/A'}
                        </strong>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Detailed View */}
        {selectedResult && (
          <div className="detailed-view">
            <div className="detailed-header">
              <div>
                <h2 className="file-title">
                  {getLanguageIcon(selectedResult.language || 'unknown')}{' '}
                  {selectedResult.filename}
                </h2>
                <p className="file-language-detail">
                  Language: <strong>{selectedResult.language}</strong>
                </p>
              </div>
            </div>

            {/* Tabs */}
            <div className="tabs">
              <button
                className={`tab-btn ${activeTab === 'code' ? 'active' : ''}`}
                onClick={() => setActiveTab('code')}
              >
                Code & Analysis
              </button>
              <button
                className={`tab-btn ${activeTab === 'metrics' ? 'active' : ''}`}
                onClick={() => setActiveTab('metrics')}
              >
                Metrics
              </button>
            </div>

            {/* Code & Highlights */}
            {activeTab === 'code' && (
              <div className="section-card">
                <h3 className="section-header">📝 Source Code</h3>
                <CodeDisplay
                  code={selectedResult.explanation.source_code || 'Source code unavailable for this result.'}
                  language={selectedResult.language || 'unknown'}
                  highlights={selectedResult.explanation.highlights}
                />
                <div className="result-details" style={{ marginTop: '12px' }}>
                  <div className="detail-row">
                    <span>Reasoning:</span>
                    <strong>{selectedResult.explanation.reasoning || 'No reasoning available.'}</strong>
                  </div>
                </div>
                {selectedResult.explanation.known_match && (
                  <div className="result-details" style={{ marginTop: '12px' }}>
                    <div className="detail-row">
                      <span>Dataset Match Type:</span>
                      <strong>{selectedResult.explanation.known_match.match_type || 'exact'}</strong>
                    </div>
                    <div className="detail-row">
                      <span>Sample Name:</span>
                      <strong>{selectedResult.explanation.known_match.filename || 'Unknown'}</strong>
                    </div>
                    <div className="detail-row">
                      <span>Source Label:</span>
                      <strong>{selectedResult.explanation.known_match.label || 'Unknown'}</strong>
                    </div>
                    <div className="detail-row">
                      <span>Match Score:</span>
                      <strong>{selectedResult.explanation.known_match.match_score || 'N/A'}</strong>
                    </div>
                    <div className="detail-row">
                      <span>Source Path:</span>
                      <strong>{selectedResult.explanation.known_match.path || 'N/A'}</strong>
                    </div>
                  </div>
                )}
                <div className="highlight-legend">
                  <div className="legend-item">
                    <div className="legend-color plagiarism" />
                    <span>
                      {selectedResult.explanation.highlight_legend?.plagiarism ||
                        'Red = segments contributing to plagiarism/similarity overlap.'}
                    </span>
                  </div>
                  <div className="legend-item">
                    <div className="legend-color ai" />
                    <span>
                      {selectedResult.explanation.highlight_legend?.['ai-detected'] ||
                        'Yellow = segments contributing to AI-like style signals.'}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Metrics */}
            {activeTab === 'metrics' && (
              <div className="section-card">
                <h3 className="section-header">📈 Detailed Metrics</h3>
                <MetricsPanel result={selectedResult} latency={latency} />
              </div>
            )}
          </div>
        )}

        {/* Export Options */}
        <div className="export-section">
          <h3>📥 Export Results</h3>
          <div className="export-buttons">
            <button className="export-btn json" onClick={handleExportJson} disabled={results.length === 0}>
              📄 JSON Report
            </button>
            <button className="export-btn csv" onClick={handleExportCsv} disabled={results.length === 0}>
              📊 CSV Export
            </button>
            <button className="export-btn pdf" onClick={handleExportPdf} disabled={results.length === 0}>
              🖨️ PDF Report
            </button>
          </div>
        </div>
      </div>
    </div>
  );
});

ResultsPage.displayName = 'ResultsPage';
export default ResultsPage;
