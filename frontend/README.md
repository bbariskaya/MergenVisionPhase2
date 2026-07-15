# MergenVision Phase 2 — Internal Demo Frontend

Professional, enterprise-style web UI for the offline video face-recognition
pipeline.

## Scope

This package is an **internal demo and validation console**. It does not touch
backend, DeepStream, or GPU worker code. Backend API contract gaps are
documented in [`docs/BACKEND_CONTRACT_GAPS.md`](./docs/BACKEND_CONTRACT_GAPS.md).

## Tech Stack

- React 18 + TypeScript
- Vite 5 + `@vitejs/plugin-react-swc`
- React Router 6
- TanStack Query v5
- Lucide React
- Vitest + React Testing Library
- Playwright

## Setup

```bash
cd frontend
npm install
```

Default development mode uses mock data. Copy `.env.example` to `.env.local`
when needed.

## Scripts

```bash
npm run dev
npm run build
npm run typecheck
npm run lint
npm run test
npm run test:e2e
```

## API / Mock Mode

`VITE_USE_MOCK_API=true` routes every request to an in-memory adapter with
deterministic fixtures. When the real backend is available, set
`VITE_USE_MOCK_API=false` and point `VITE_API_BASE_URL` to the running API.

## Routes

- `/` — Dashboard
- `/videos/new` — Upload a new video
- `/videos/jobs/:jobId` — Job progress and result
- `/faces/:faceId` — Face appearances across videos
