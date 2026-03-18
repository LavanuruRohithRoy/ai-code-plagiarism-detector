import React, { useState, useRef, memo } from 'react';
import './Upload.css';
import { useAnalysis } from '../hooks/useAnalysis';
import { SUPPORTED_LANGUAGES, getLanguageIcon } from '../utils/constants';
import LoadingSpinner from '../components/LoadingSpinner';

interface UploadPageProps {
  onResultsReady: (data: any) => void;
}

const UploadPage = memo<UploadPageProps>(({ onResultsReady }) => {
  const [files, setFiles] = useState<File[]>([]);
  const [mode, setMode] = useState<'single' | 'batch'>('single');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { loading, error, analyzeFiles, analyzeFile, reset } = useAnalysis();

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    const droppedFiles = Array.from(e.dataTransfer.files).filter(
      (f) => f.type.startsWith('text/') || f.name.match(/\.(py|js|jsx|ts|tsx|java|c|cpp|go|rs)$/i)
    );
    setFiles((prev) => [...prev, ...droppedFiles]);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    setFiles((prev) => [...prev, ...selectedFiles]);
  };

  const handleRemoveFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleAnalyze = async () => {
    if (files.length === 0) return;

    try {
      if (mode === 'single' && files.length === 1) {
        const result = await analyzeFile(files[0]);
        onResultsReady({ single: result, latency: (result as any)._latency });
      } else {
        const result = await analyzeFiles(files);
        onResultsReady({ batch: result, latency: result._latency });
      }
    } catch (err) {
      console.error('Analysis failed:', err);
    }
  };

  const handleClear = () => {
    setFiles([]);
    reset();
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="upload-page">
      <div className="upload-container">
        {/* Header */}
        <div className="upload-header">
          <div className="header-content">
            <h1 className="title">AI Code Plagiarism Detector</h1>
            <p className="subtitle">
              Upload your code files to detect plagiarism and AI-generated content
            </p>
          </div>
        </div>

        {/* Mode Selector */}
        <div className="mode-selector">
          <button
            className={`mode-btn ${mode === 'single' ? 'active' : ''}`}
            onClick={() => {
              setMode('single');
              setFiles(files.slice(0, 1));
            }}
          >
            📄 Single File
          </button>
          <button
            className={`mode-btn ${mode === 'batch' ? 'active' : ''}`}
            onClick={() => setMode('batch')}
          >
            📦 Batch Upload
          </button>
        </div>

        {/* Drop Zone */}
        <div
          className={`drop-zone ${files.length > 0 ? 'has-files' : ''}`}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <div className="drop-icon">📁</div>
          <h2 className="drop-title">Drag & drop your code files here</h2>
          <p className="drop-subtitle">or click the button below to browse</p>
          <button
            className="browse-btn"
            onClick={(e) => {
              e.stopPropagation();
              fileInputRef.current?.click();
            }}
          >
            Browse Files
          </button>
          <input
            ref={fileInputRef}
            type="file"
            multiple={mode === 'batch'}
            accept=".py,.js,.ts,.java,.c,.cpp,.go,.rs,.jsx,.tsx"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />
          <p className="supported-text">
            Supported: Python, JavaScript, TypeScript, Java, C, C++, Go, Rust
          </p>
        </div>

        {/* File List */}
        {files.length > 0 && (
          <div className="files-section">
            <h3 className="section-title">
              {mode === 'single' ? 'Selected File' : `Selected Files (${files.length})`}
            </h3>
            <div className="files-list">
              {files.map((file, idx) => {
                const ext = file.name.split('.').pop() || 'unknown';
                const extensionLanguageMap: Record<string, string> = {
                  py: 'python',
                  js: 'javascript',
                  jsx: 'javascript',
                  ts: 'typescript',
                  tsx: 'typescript',
                  java: 'java',
                  c: 'c',
                  cpp: 'cpp',
                  go: 'go',
                  rs: 'rust',
                };
                const normalizedExt = ext.toLowerCase();
                const language = extensionLanguageMap[normalizedExt] ||
                  SUPPORTED_LANGUAGES.find((lang) =>
                    file.name.toLowerCase().endsWith(
                      lang === 'cpp' ? '.cpp' : lang === 'typescript' ? '.ts' : `.${lang}`
                    )
                  ) || normalizedExt;

                return (
                  <div key={`${file.name}-${idx}`} className="file-item">
                    <span className="file-icon">{getLanguageIcon(language)}</span>
                    <div className="file-info">
                      <div className="file-name">{file.name}</div>
                      <div className="file-meta">
                        {(file.size / 1024).toFixed(2)} KB •{' '}
                        <span className="language-tag">{language}</span>
                      </div>
                    </div>
                    <button
                      className="remove-btn"
                      onClick={() => handleRemoveFile(idx)}
                      aria-label="Remove file"
                    >
                      ✕
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="error-banner">
            <span className="error-icon">⚠️</span>
            <p className="error-text">{error}</p>
            <button
              className="error-close"
              onClick={() => handleClear()}
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Action Buttons */}
        <div className="action-buttons">
          {loading ? (
            <LoadingSpinner
              message={`Analyzing ${files.length} code ${files.length === 1 ? 'file' : 'files'}...`}
            />
          ) : (
            <>
              <button
                className="analyze-btn"
                onClick={handleAnalyze}
                disabled={files.length === 0}
              >
                🔍 Analyze Code
              </button>
              {files.length > 0 && (
                <button className="clear-btn" onClick={handleClear}>
                  Clear All
                </button>
              )}
            </>
          )}
        </div>

        {/* Info Section */}
        <div className="info-section">
          <div className="info-card">
            <h4>How It Works</h4>
            <ol>
              <li>Upload one or more code files</li>
              <li>Our detector analyzes the code for plagiarism</li>
              <li>Get detailed results with highlighted similarities</li>
            </ol>
          </div>

          <div className="info-card">
            <h4>What We Detect</h4>
            <ul>
              <li>Plagiarism from known datasets</li>
              <li>AI-generated code patterns</li>
              <li>Code structure and similarity metrics</li>
              <li>Token and semantic matches</li>
            </ul>
          </div>

          <div className="info-card">
            <h4>Privacy & Security</h4>
            <ul>
              <li>Files analyzed locally on server</li>
              <li>No permanent storage of code</li>
              <li>Results returned immediately</li>
              <li>Enterprise-grade encryption</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
});

UploadPage.displayName = 'UploadPage';
export default UploadPage;
