import React, { useState } from 'react';
import './App.css';
import UploadPage from './pages/Upload';
import ResultsPage from './pages/Results';

type AppPage = 'upload' | 'results';

interface ResultsData {
  single?: any;
  batch?: any;
  latency?: number;
}

function App() {
  const [currentPage, setCurrentPage] = useState<AppPage>('upload');
  const [resultsData, setResultsData] = useState<ResultsData | null>(null);

  const handleResultsReady = (data: ResultsData) => {
    setResultsData(data);
    setCurrentPage('results');
  };

  const handleBackToUpload = () => {
    setCurrentPage('upload');
    setResultsData(null);
  };

  return (
    <div className="app">
      {currentPage === 'upload' ? (
        <UploadPage onResultsReady={handleResultsReady} />
      ) : (
        <ResultsPage data={resultsData!} onBack={handleBackToUpload} />
      )}
    </div>
  );
}

export default App;
