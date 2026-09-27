# ColorRevive Frontend (Next.js + TypeScript + Tailwind)

Photo-studio-inspired single-page workspace: drag-and-drop upload, settings,
processing states, before/after comparison slider, downloads, history, dark
mode.

## Run locally

```bash
npm ci
cp ../.env.example .env.local   # set NEXT_PUBLIC_API_BASE_URL (default http://localhost:8000)
npm run dev                     # http://localhost:3000
```

## Scripts

| Command | What |
|---|---|
| `npm run dev` | Next.js dev server (:3000) |
| `npm run build` | Production build (standalone output for Docker/Vercel) |
| `npm start` | Serve the production build |
| `npm test` | Vitest + React Testing Library (28 tests) |
| `npm run lint` | ESLint (next/core-web-vitals) |
| `npm run format` | Prettier over app/components/lib/tests |
| `npm run e2e` | Playwright happy-path test (starts both servers if configured) |

## Structure

```
app/page.tsx          # hero, workspace, how-it-works, limitations, footer
components/
  upload-zone.tsx     # react-dropzone, MIME+extension validation, keyboard accessible
  image-preview.tsx   # aspect-preserving preview w/ dimensions & size
  settings-panel.tsx  # quality, preserve contrast, denoise, output format
  processing-state.tsx# staged progress + aria-live announcements
  comparison-slider.tsx # pointer/touch/keyboard draggable divider (ARIA slider)
  result-actions.tsx  # PNG/JPEG download, start over
  history-panel.tsx   # localStorage session history (thumbnails stay client-side)
  theme-toggle.tsx    # light/dark respecting prefers-color-scheme/reduced-motion
lib/api.ts            # fetch wrapper: timeouts, typed errors, base64 decode
lib/types.ts          # shared TS types mirroring backend schemas
lib/utils.ts          # formatting helpers, file validators
tests/                # Vitest suites per component + utils
```

## Behavior contract with the API

- The banner reads `fallback_mode` from `/api/v1/info`; when true a persistent
  amber badge shows **“Fallback mode — deterministic demo coloring, not AI”**
  so results are never misrepresented.
- Duplicate submissions are blocked while `processing`.
- Errors map to human messages (unsupported type, too large, corrupt, network,
  backend down, timeout); stack traces are never shown.
- Downloads use the actual returned bytes named
  `colorrevive-<original>-colorized.<ext>`.

## Accessibility

WCAG 2.1 AA focus: semantic landmarks, labelled controls (no placeholder-as-label),
visible focus rings, ARIA live status region, keyboard-operable slider
(`role="slider"`, arrow keys ±1 %, Home/End), reduced-motion support, ≥4.5:1 text
contrast in both themes.
