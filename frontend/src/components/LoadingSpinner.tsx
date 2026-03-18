import React, { memo } from 'react';
import './LoadingSpinner.css';

interface LoadingSpinnerProps {
  message?: string;
}

const LoadingSpinner = memo<LoadingSpinnerProps>(
  ({ message = 'Analyzing code...' }) => (
    <div className="loading-container">
      <div className="spinner">
        <div className="spinner-ring" />
        <div className="spinner-ring" />
        <div className="spinner-ring" />
      </div>
      <p className="loading-message">{message}</p>
    </div>
  )
);

LoadingSpinner.displayName = 'LoadingSpinner';
export default LoadingSpinner;
