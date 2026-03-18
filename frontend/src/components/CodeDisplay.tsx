import React, { memo } from 'react';
import './CodeDisplay.css';
import { Highlight } from '../api/client';
import { getHighlightColor } from '../utils/codeHighlight';

interface CodeDisplayProps {
  code: string;
  language: string;
  highlights?: Highlight[];
  lineNumbers?: boolean;
}

const CodeDisplay = memo<CodeDisplayProps>(
  ({ code, language, highlights = [], lineNumbers = true }) => {
    const lines = code.split('\n');

    const getHighlightStyle = (charIndex: number): React.CSSProperties => {
      const highlight = highlights.find(
        (h) => charIndex >= h.start && charIndex < h.end
      );
      if (highlight) {
        return {
          backgroundColor: getHighlightColor(highlight.type),
          color: 'inherit',
        };
      }
      return {};
    };

    return (
      <div className="code-display">
        <div className="code-header">
          <span className="code-language">{language.toUpperCase()}</span>
          <span className="code-lines">{lines.length} lines</span>
        </div>
        <pre className="code-container">
          <code>
            {lines.map((line, lineIdx) => {
              let charOffset = lines
                .slice(0, lineIdx)
                .reduce((sum, l) => sum + l.length + 1, 0); // +1 for newline

              return (
                <div key={lineIdx} className="code-line">
                  {lineNumbers && (
                    <span className="line-number">{lineIdx + 1}</span>
                  )}
                  <span className="line-content">
                    {line.split('').map((char, charIdx) => {
                      const style = getHighlightStyle(charOffset + charIdx);
                      return (
                        <span
                          key={`${lineIdx}-${charIdx}`}
                          style={style}
                          className="code-char"
                        >
                          {char}
                        </span>
                      );
                    })}
                  </span>
                </div>
              );
            })}
          </code>
        </pre>
      </div>
    );
  }
);

CodeDisplay.displayName = 'CodeDisplay';
export default CodeDisplay;
