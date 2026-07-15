# MergenVision Phase 2 — Internal Demo Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILLS:
> - `superpowers:test-driven-development` for pure utilities.
> - `superpowers:verification-before-completion` before all claims.
> - `superpowers:context7-mcp` for library behavior confirmation.

**Goal:** Build a professional, enterprise-style internal web UI that
visualises the offline video face-recognition pipeline: upload → async job →
progress → canonical person results → timeline → video + bbox overlay.

**Architecture:**
Single-page React + Vite + TypeScript app. `frontend/src/api` isolates all
backend communication behind typed contracts and a mock adapter. React Router
v6 handles `/`, `/videos/new`, `/videos/jobs/:jobId`, `/faces/:faceId`. Global
design tokens live in `frontend/src/styles/tokens.css` and are consumed by
vanilla CSS modules. Heavy rendering (canvas bbox overlay, timeline) is
memoised; no DOM-per-bbox approach. Playwright provides mock E2E coverage and
screenshots.

**Tech Stack:**
React 18, Vite 5, TypeScript 5, React Router 6, TanStack Query v5,
Lucide React, Vitest + React Testing Library, Playwright.

## Global Constraints

- Only files under `frontend/` are created or modified.
- No commits, pushes, or destructive git operations.
- `VITE_USE_MOCK_API=true` in dev; mock fixtures never auto-activate in
  production.
- Design tokens in CSS variables; light theme only for this sprint.
- Bbox overlay must use canvas with correct letterbox math (ResizeObserver +
  devicePixelRatio).
- Raw cosine similarity must never be rendered as a percentage confidence.
- Backend contract gaps are documented in
  `frontend/docs/BACKEND_CONTRACT_GAPS.md`.

---

## File Structure

```text
frontend/
├── .env.example
├── .gitignore
├── docs/
│   └── BACKEND_CONTRACT_GAPS.md
├── index.html
├── package.json
├── playwright.config.ts
├── public/
│   └── mock-videos/
│       └── friendsshort.mp4   (symlink or small fixture)
├── README.md
├── tsconfig.json
├── vite.config.ts
└── src/
    ├── main.tsx
    ├── App.tsx
    ├── App.css
    ├── routes.tsx
    ├── styles/
    │   ├── tokens.css
    │   ├── global.css
    │   └── utilities.css
    ├── api/
    │   ├── client.ts
    │   ├── contracts.ts
    │   ├── errors.ts
    │   ├── videos.ts
    │   ├── jobs.ts
    │   ├── faces.ts
    │   └── mock/
    │       ├── index.ts
    │       ├── fixtures.ts
    │       └── progressSimulator.ts
    ├── components/
    │   ├── layout/
    │   │   ├── AppLayout.tsx
    │   │   ├── Sidebar.tsx
    │   │   └── TopBar.tsx
    │   ├── ui/
    │   │   ├── Button.tsx
    │   │   ├── Card.tsx
    │   │   ├── Dialog.tsx
    │   │   ├── Input.tsx
    │   │   ├── Select.tsx
    │   │   ├── Skeleton.tsx
    │   │   ├── StatusBadge.tsx
    │   │   ├── EmptyState.tsx
    │   │   └── ErrorState.tsx
    │   ├── dashboard/
    │   │   ├── DashboardPage.tsx
    │   │   ├── JobsTable.tsx
    │   │   └── MetricCards.tsx
    │   ├── upload/
    │   │   ├── UploadPage.tsx
    │   │   ├── DropZone.tsx
    │   │   └── UploadForm.tsx
    │   ├── job/
    │   │   ├── JobPage.tsx
    │   │   ├── ProgressStepper.tsx
    │   │   ├── JobSummary.tsx
    │   │   └── CancelDialog.tsx
    │   └── result/
    │       ├── ResultPage.tsx
    │       ├── VideoPlayer.tsx
    │       ├── BboxCanvas.tsx
    │       ├── PersonList.tsx
    │       ├── PersonCard.tsx
    │       ├── Timeline.tsx
    │       ├── TimelineRow.tsx
    │       └── TechnicalDetails.tsx
    ├── faces/
    │   └── FaceAppearancesPage.tsx
    ├── hooks/
    │   ├── useJobProgress.ts
    │   └── useVideoContainer.ts
    ├── lib/
    │   ├── bboxTransform.ts
    │   ├── detectionLookup.ts
    │   ├── durationFormat.ts
    │   ├── identityStatus.ts
    │   ├── statusMapping.ts
    │   ├── timeline.ts
    │   └── types.ts
    └── test/
        ├── setup.ts
        ├── mocks/
        └── e2e/
            └── demo.spec.ts
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tsconfig.json`
- Create: `frontend/index.html`
- Create: `frontend/.env.example`
- Create: `frontend/README.md`
- Create: `frontend/.gitignore`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/routes.tsx`
- Create: `frontend/public/mock-videos/`

**Commands:**

Run from repository root:

```bash
cd frontend
npm create vite@latest . -- --template react-ts
npm install react-router-dom @tanstack/react-query lucide-react
npm install -D vitest @testing-library/react @testing-library/jest-dom @testing-library/user-event jsdom @playwright/test @types/node
```

**Verification:**

```bash
npm run dev &
curl -s http://localhost:5173 | grep -q "Vite" && echo "dev server up"
```

- [ ] Vite dev server starts with default React template.
- [ ] `npm run build` exits 0.
- [ ] `npm run test` runs Vitest.

---

## Task 2: Design Tokens and Layout

**Files:**
- Create: `frontend/src/styles/tokens.css`
- Create: `frontend/src/styles/global.css`
- Create: `frontend/src/styles/utilities.css`
- Create: `frontend/src/components/layout/AppLayout.tsx`
- Create: `frontend/src/components/layout/Sidebar.tsx`
- Create: `frontend/src/components/layout/TopBar.tsx`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/routes.tsx`

**Required tokens (excerpt):**

```css
:root {
  --color-bg: #f7f8fa;
  --color-surface: #ffffff;
  --color-surface-subtle: #f0f2f5;
  --color-border: #e1e4e8;
  --color-text: #111827;
  --color-text-muted: #5f6b7a;
  --color-primary: #1e5aa8;
  --color-primary-hover: #174885;
  --color-success: #1a7f45;
  --color-warning: #b35900;
  --color-danger: #c62828;
  --radius-sm: 4px;
  --radius-md: 6px;
  --radius-lg: 8px;
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.04);
  --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.06);
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 20px;
  --space-6: 24px;
  --space-8: 32px;
}
```

**Verification:**

- [ ] Sidebar renders four navigation items.
- [ ] Active route is highlighted.
- [ ] Layout is usable at 1280×720 and 1920×1080.
- [ ] No horizontal scroll on dashboard.

---

## Task 3: API Contracts and Mock Adapter

**Files:**
- Create: `frontend/src/api/contracts.ts`
- Create: `frontend/src/api/errors.ts`
- Create: `frontend/src/api/client.ts`
- Create: `frontend/src/api/videos.ts`
- Create: `frontend/src/api/jobs.ts`
- Create: `frontend/src/api/faces.ts`
- Create: `frontend/src/api/mock/index.ts`
- Create: `frontend/src/api/mock/fixtures.ts`
- Create: `frontend/src/api/mock/progressSimulator.ts`

**Contracts (excerpt in `frontend/src/api/contracts.ts`):**

```ts
export interface VideoJob {
  jobId: string;
  processId: string;
  status: VideoJobStatus;
  createdAt: string;
  videoName?: string;
  video?: VideoMetadata;
  progress?: JobProgress;
  result?: VideoResult;
  error?: ApiErrorBody;
}

export type VideoJobStatus =
  | 'pending'
  | 'uploading'
  | 'validating'
  | 'queued'
  | 'processing'
  | 'finalizing'
  | 'rendering'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface CanonicalPerson {
  faceId: string;
  trackId: string;
  status: 'known' | 'anonymous' | 'new_anonymous' | 'unknown';
  name: string | null;
  metadata: Record<string, unknown>;
  firstSeen: number;
  lastSeen: number;
  totalDuration: number;
  confidence?: number;
  similarity?: number;
  margin?: number;
  appearances: AppearanceInterval[];
  detections: FrameDetection[];
}
```

**Mock adapter:**

- Returns deterministic fixtures for `friendsshort.mp4`.
- Simulates progress from `uploading` to `completed` over ~8 seconds when no
  fixture id is pre-set.
- Supports `DELETE` to cancel a processing job.
- Exposes `RESET_MOCK_DATA()` helper for tests.

**Verification:**

- [ ] `videos.recognize(file)` returns a job id.
- [ ] `jobs.get(jobId)` returns fixture data.
- [ ] Polling via TanStack Query drains the progress simulator.
- [ ] Cancel sets status to `cancelled`.

---

## Task 4: Pure Utilities (TDD)

**Files:**
- Create: `frontend/src/lib/bboxTransform.ts` + `.test.ts`
- Create: `frontend/src/lib/detectionLookup.ts` + `.test.ts`
- Create: `frontend/src/lib/durationFormat.ts` + `.test.ts`
- Create: `frontend/src/lib/statusMapping.ts` + `.test.ts`
- Create: `frontend/src/lib/identityStatus.ts` + `.test.ts`
- Create: `frontend/src/lib/timeline.ts` + `.test.ts`

**Key functions:**

```ts
// bboxTransform.ts
export function computeLetterbox(
  sourceWidth: number,
  sourceHeight: number,
  containerWidth: number,
  containerHeight: number,
): { scale: number; offsetX: number; offsetY: number } { ... }

export function sourceToDisplay(
  box: BoundingBox,
  letterbox: ReturnType<typeof computeLetterbox>,
): DisplayBoundingBox { ... }

// detectionLookup.ts
export function findDetectionForTime(
  detections: FrameDetection[],
  time: number,
): FrameDetection[] { ... }
```

**TDD cycle:**

- [ ] RED: write each failing test.
- [ ] GREEN: implement minimal function.
- [ ] REFACTOR: simplify.
- [ ] All unit tests pass.

---

## Task 5: Dashboard

**Files:**
- Create: `frontend/src/components/dashboard/DashboardPage.tsx`
- Create: `frontend/src/components/dashboard/JobsTable.tsx`
- Create: `frontend/src/components/dashboard/MetricCards.tsx`
- Modify: `frontend/src/routes.tsx`

**Behaviors:**

- Lists latest jobs with status badges.
- Metrics are shown only if backend returns data; mock returns deterministic
  numbers.
- Empty, loading skeleton, and error retry states.
- "Yeni Video Analizi" CTA.

**Verification:**

- [ ] Dashboard renders table with fixture jobs.
- [ ] Status badges use color + text + icon.
- [ ] Error state has retry button.
- [ ] Empty state visible when no jobs.

---

## Task 6: Upload Form

**Files:**
- Create: `frontend/src/components/upload/UploadPage.tsx`
- Create: `frontend/src/components/upload/DropZone.tsx`
- Create: `frontend/src/components/upload/UploadForm.tsx`

**Behaviors:**

- Drag-and-drop zone with keyboard support.
- File type/size validation.
- Form fields: sampling rate, minimum face size, profile (Accuracy/Balanced/Fast).
- Upload progress simulation.
- Cancel and duplicate-submit prevention.
- On success, navigate to `/videos/jobs/:jobId`.

**Verification:**

- [ ] DropZone accepts `.mp4`, `.mov`, `.avi`.
- [ ] Oversize file shows validation error.
- [ ] Submit navigates to job page.
- [ ] Double submit is blocked.

---

## Task 7: Job Progress

**Files:**
- Create: `frontend/src/components/job/JobPage.tsx`
- Create: `frontend/src/components/job/ProgressStepper.tsx`
- Create: `frontend/src/components/job/JobSummary.tsx`
- Create: `frontend/src/components/job/CancelDialog.tsx`
- Create: `frontend/src/hooks/useJobProgress.ts`

**Behaviors:**

- Stepper maps backend status to visible stages.
- Polls every 2 s while processing.
- Shows processed frames, detected faces, elapsed time.
- Cancel button with confirmation dialog.
- Completed job auto-offers link to result.

**Verification:**

- [ ] Processing job reaches completed after simulation.
- [ ] Cancel shows confirmation, then cancelled state.
- [ ] Stepper advances.

---

## Task 8: Result Page

**Files:**
- Create: `frontend/src/components/result/ResultPage.tsx`
- Create: `frontend/src/components/result/VideoPlayer.tsx`
- Create: `frontend/src/components/result/BboxCanvas.tsx`
- Create: `frontend/src/components/result/PersonList.tsx`
- Create: `frontend/src/components/result/PersonCard.tsx`
- Create: `frontend/src/components/result/Timeline.tsx`
- Create: `frontend/src/components/result/TimelineRow.tsx`
- Create: `frontend/src/components/result/TechnicalDetails.tsx`
- Create: `frontend/src/hooks/useVideoContainer.ts`

**Behaviors:**

- Two-pane layout: video + canvas on the left, person list on the right.
- Canvas overlay draws original-resolution bboxes with letterbox math.
- Person selection seeks video to first appearance.
- Timeline rows per canonical person with clickable segments.
- Known / anonymous / new_anonymous / unknown visual distinction.
- Raw similarity vs calibrated confidence separated.
- Technical details accordion with tracklet mappings.
- JSON result download button (mock only if backend lacks endpoint).

**Verification:**

- [ ] Bbox aligned with video content under resize.
- [ ] Clicking person row seeks video.
- [ ] Timeline segment click seeks video.
- [ ] No-face completed shows success empty state.
- [ ] Failed job shows sanitized error.

---

## Task 9: Face Appearances Page

**Files:**
- Create: `frontend/src/faces/FaceAppearancesPage.tsx`

**Behaviors:**

- Loads `GET /faces/{faceId}/appearances` via mock.
- Lists videos/jobs where the face appears with timestamps.
- Link to job result with seek.

**Verification:**
- [ ] Known face shows name.
- [ ] Unknown face shows id.
- [ ] Each appearance links to `/videos/jobs/:jobId`.

---

## Task 10: E2E Tests and Screenshots

**Files:**
- Create: `frontend/playwright.config.ts`
- Create: `frontend/src/test/e2e/demo.spec.ts`

**Screenshots to generate:**

```text
frontend/artifacts/screenshots/
├── dashboard-1440.png
├── upload-1440.png
├── processing-1440.png
├── result-1440.png
├── result-person-selected-1440.png
├── result-1280.png
├── no-face.png
└── failed.png
```

**Verification:**

- [ ] `npm run test:e2e` passes.
- [ ] Screenshots are visually reviewed for clipping, overflow, alignment.
- [ ] `frontend/artifacts/` is added to `.gitignore`.

---

## Task 11: Final Verification

Run all commands fresh and report:

```bash
npm ci
npm run lint
npm run typecheck
npm run test
npm run build
npm run test:e2e
git status --short
git diff --check
```

Claim completion only after collecting evidence.
