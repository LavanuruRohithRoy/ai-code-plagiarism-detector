import React, { memo } from 'react';
import './MetricsPanel.css';
import { AnalysisResult } from '../api/client';
import {
  formatPercentage,
  formatLatency,
  getConfidenceColor,
  normalizeScoreToPercent,
} from '../utils/constants';

interface MetricsPanelProps {
  result: AnalysisResult;
  latency?: number;
}

const MetricsPanel = memo<MetricsPanelProps>(({ result, latency }) => {
  const { plagiarism_percentage, ai_probability, confidence, explanation } =
    result;
  const confidenceColor = getConfidenceColor(confidence);
  const plagiarismPercent = normalizeScoreToPercent(plagiarism_percentage);
  const aiPercent = normalizeScoreToPercent(ai_probability);

  return (
    <div className="metrics-panel">
      <div className="metrics-grid">
        {/* Plagiarism Percentage */}
        <div className="metric-card">
          <div className="metric-header">
            <h3 className="metric-title">Plagiarism Detected</h3>
            <div className="metric-icon plagiarism">⚠️</div>
          </div>
          <div className="metric-value plagiarism-value">
            {formatPercentage(plagiarism_percentage)}%
          </div>
          <div className="metric-bar">
            <div
              className="metric-bar-fill plagiarism"
              style={{
                width: `${Math.min(plagiarismPercent, 100)}%`,
              }}
            />
          </div>
          <div className="metric-description">
            {plagiarismPercent > 70
              ? 'High plagiarism risk'
              : plagiarismPercent > 30
              ? 'Moderate plagiarism detected'
              : 'Low plagiarism risk'}
          </div>
        </div>

        {/* AI Probability */}
        <div className="metric-card">
          <div className="metric-header">
            <h3 className="metric-title">AI Generation Probability</h3>
            <div className="metric-icon ai">🤖</div>
          </div>
          <div className="metric-value ai-value">
            {formatPercentage(ai_probability)}%
          </div>
          <div className="metric-bar">
            <div
              className="metric-bar-fill ai"
              style={{
                width: `${Math.min(aiPercent, 100)}%`,
              }}
            />
          </div>
          <div className="metric-description">
            {aiPercent > 70
              ? 'Likely AI-generated'
              : aiPercent > 30
              ? 'Possibly AI-generated'
              : 'Likely human-written'}
          </div>
        </div>

        {/* Confidence Level */}
        <div className="metric-card">
          <div className="metric-header">
            <h3 className="metric-title">Confidence Level</h3>
            <div className="metric-icon" style={{ color: confidenceColor }}>
              ✓
            </div>
          </div>
          <div
            className="metric-value confidence-value"
            style={{ color: confidenceColor }}
          >
            {confidence.toUpperCase()}
          </div>
          <div
            className="confidence-badge"
            style={{
              borderColor: confidenceColor,
              color: confidenceColor,
            }}
          >
            {confidence === 'high'
              ? 'Highly Confident'
              : confidence === 'medium'
              ? 'Moderately Confident'
              : 'Low Confidence'}
          </div>
        </div>
      </div>

      {/* Detailed Explanation */}
      <div className="explanation-section">
        <h3 className="section-title">Analysis Details</h3>

        <div className="details-grid">
          {explanation.code_lines && (
            <div className="detail-item">
              <label>Lines of Code</label>
              <span>{explanation.code_lines}</span>
            </div>
          )}
          {explanation.token_similarity !== undefined && (
            <div className="detail-item">
              <label>Token Similarity</label>
              <span>{formatPercentage(explanation.token_similarity)}%</span>
            </div>
          )}
          {explanation.semantic_similarity !== undefined && (
            <div className="detail-item">
              <label>Semantic Similarity</label>
              <span>
                {formatPercentage(explanation.semantic_similarity)}%
              </span>
            </div>
          )}
          {explanation.structure_similarity !== undefined && (
            <div className="detail-item">
              <label>Structure Similarity</label>
              <span>
                {formatPercentage(explanation.structure_similarity)}%
              </span>
            </div>
          )}
          {explanation.language && (
            <div className="detail-item">
              <label>Detected Language</label>
              <span>{explanation.language}</span>
            </div>
          )}
          {explanation.size_penalty_applied !== undefined && (
            <div className="detail-item">
              <label>Size Penalty Applied</label>
              <span>{explanation.size_penalty_applied ? 'Yes' : 'No'}</span>
            </div>
          )}
        </div>

        {/* Reasoning */}
        <div className="reasoning">
          <h4>Reasoning</h4>
          <p>{explanation.reasoning}</p>
        </div>
      </div>

      {/* Performance Metrics */}
      {latency !== undefined && (
        <div className="performance-section">
          <div className="performance-metric">
            <span className="perf-label">Analysis Time</span>
            <span className="perf-value">{formatLatency(latency)}</span>
          </div>
        </div>
      )}
    </div>
  );
});

MetricsPanel.displayName = 'MetricsPanel';
export default MetricsPanel;
