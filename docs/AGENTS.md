# Cognitive Engine — Agent Notes

## Layout

- Implementation in TypeScript (`tests/*.test.ts` run by jest; package
  scripts in `package.json`, e.g. `npm test`, `test:fractal`, `test:mmm`).
  Python/TS hybrid elements noted in `setup.py` and `requirements.txt` at
  repo root.
- `docs/` mirrors module names: `fractal.md`, `probabilistic.md`,
  `memory.md`, `unipixel.md`, `mmm.md`, `code_execution.md`, `peff.md`;
  API summaries under `docs/api/`.

## Conventions observed

- Plain Markdown relative links (no wiki-links).
- `index.md` files serve as per-directory tables of contents.

## Maintenance rules

- When adding a module doc, link it from `docs/index.md` and
  `docs/README.md`.
- Do not link files that do not exist; verify targets when editing
  (docs-audit 2026-08-29 repaired 21 dangling links in `index.md`,
  `api/index.md`, `neural_network_testing.md`, and `contributing.md`).

## Commands (from package.json)

```bash
npm test          # jest --verbose
npm run test:all  # jest tests/*.test.ts --verbose
```
