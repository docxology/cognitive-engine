# AGENTS.md — cognitive-engine

Guidance for agents working in this repository.

## What this repository is

A hybrid neuro-symbolic AI framework with two pathways over persistent memory:

- a **seven-level fractal symbolic reasoner** (`cognitive_engine/fractal/`), and
- a **probabilistic path** built on PyTorch/Transformers
  (`cognitive_engine/probabilistic/`, `cognitive_engine/llm/`).

Around them sit modules for memory, unipixel encoding, MMM cross-layer pattern
mining, code execution, and PEFF ethics/security. `cognitive_engine/engine.py`
is the composition point.

It is published as the Python package `cognitive_engine` (version 0.1.0 via
`setup.py` + `requirements.txt`) **and** carries a TypeScript/Jest test layer
declared in `package.json`. Both exist; neither is optional.

## Layout

| Path | Role |
|---|---|
| `cognitive_engine/engine.py` | Top-level engine composition |
| `cognitive_engine/fractal/` | Seven-level symbolic reasoner |
| `cognitive_engine/probabilistic/` | PyTorch/Transformers pathway |
| `cognitive_engine/llm/` | Model interface |
| `cognitive_engine/memory/` | Persistent memory |
| `cognitive_engine/unipixel/` | Unipixel representation |
| `cognitive_engine/mmm/` | Cross-layer pattern mining |
| `cognitive_engine/code_execution/` | Sandboxed code execution |
| `cognitive_engine/peff/` | Ethics and security checks |
| `tests/` | Jest suites (`*.test.ts`) plus `runner.ts`, `run_neural_tests.*` |
| `run_tests.py`, `run_tests.sh`, `run_neural_tests.py` | Python test entry points |
| `models/` | Model artefacts |
| `docs/` | Documentation hub |
| `coverage/` | Committed coverage output — see the note below |

## Running tests

Two suites, two runners:

```bash
npm test                    # jest, all *.test.ts
npm run test:fractal        # single module
python3 run_tests.py        # Python-side runner
```

Jest per-module scripts exist for `fractal`, `probabilistic`, `memory`,
`unipixel`, `mmm`, `codeexec`, and `peff`. Use the per-module script when you
touch one module.

## Conventions

- `node_modules/` is not tracked. Install with `npm install` before running Jest.
  Bytecode and virtualenvs are likewise ignored via `.gitignore`.
- `coverage/` is currently tracked build output. It is not a source of truth;
  regenerate it rather than reading numbers out of it, and prefer untracking it
  over editing it.
- The fractal reasoner and the probabilistic pathway must stay separable. A
  change to one should not force a dependency from the other; `engine.py` is
  where they meet.
- `code_execution/` runs code. Treat any widening of what it will execute as a
  security change, not a feature.
- `peff/` encodes ethics and security policy. Do not weaken a check to make an
  example pass; fix the example.
- `models/` holds binary artefacts. Adding one increases clone size — prefer a
  release asset when the file is large.

## Verification

Run `npm test` and the Python runner for the area you changed, and report the
exact commands and results. There is no CI workflow in this repository, so a
green claim is only as good as the run you performed.
