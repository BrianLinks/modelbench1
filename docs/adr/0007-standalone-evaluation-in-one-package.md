# 7. One standalone package, with evaluation built in

Status: accepted

## Context
Tracking runs and judging them are different jobs, and a project could split them into two
packages that depend on each other. For a tool meant to be downloaded, run offline and read in
one sitting, that split has costs: two things to install and version, a shared format that must
be kept in step, and results that cannot be checked without the second package.

## Decision
- ModelBench is one package that stands alone. It imports nothing from any sibling project and
  has no runtime dependencies (see ADR 1).
- The statistics needed to judge runs live inside it: significance tests (`stats`), ROC and
  precision-recall curves, calibration, permutation importance, learning curves, a paired
  bootstrap, and a standalone `report`.
- Other tools can exchange results through the versioned export format (ADR 5) and nothing else.
  The format is frozen by a golden file, so a reader written against version 1 keeps working.
- No feature may need a paid service or an account. Everything uses local files and seeded
  generators.

## Consequences
- The package is larger than a bare tracker, so each addition has to earn its place: a feature
  is added only with known-answer tests and a mutation check, and the changelog says what it does.
- A tool that wants to build on ModelBench reads exports instead of importing its internals.
- Splitting a module out later stays possible, because layers only depend downwards
  (see `docs/ARCHITECTURE.md`).
