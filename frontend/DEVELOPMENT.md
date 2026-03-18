# Frontend Development

React + TypeScript frontend application.

See the root [README.md](../README.md) for full setup and configuration.

## Quick Commands

```bash
npm install    # Install dependencies
npm run dev    # Start development server (port 3000)
npm run build  # Build for production
```

## Structure

```
src/
├── pages/      # Upload & Results pages
├── components/ # Reusable UI components
├── api/        # API client
├── hooks/      # Custom React hooks
├── utils/      # Utilities
└── styles/     # CSS & design tokens
```

## Key Files

- **api/client.ts**: Axios API client for backend communication
- **hooks/useAnalysis.ts**: State management for analysis
- **components/CodeDisplay.tsx**: Code viewer with highlights
- **components/MetricsPanel.tsx**: Metrics visualization
- **styles/variables.css**: Design tokens and CSS variables

## ## Configuration

Set `VITE_API_BASE_URL` in `.env.local` to point to your backend:

```
VITE_API_BASE_URL=http://localhost:8000
```

## Troubleshooting

- **Port in use**: `npm run dev -- --port 3001`
- **API errors**: Ensure backend is running on configured URL
- **Cache issues**: Clear browser cache (Ctrl+Shift+Delete)
