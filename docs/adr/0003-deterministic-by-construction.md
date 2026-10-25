# 3. Deterministic by construction

Status: accepted

## Context
A comparison between two runs is meaningless if the difference could be a different random
split or a different dataset.

## Decision
- Sub-seeds come from SHA-256 of the experiment seed and a purpose string, never from global
  random state.
- Dataset versions are a hash of the content, so a changed file is a new version.
- The config hash covers every setting that affects results and ignores the experiment label.
- Each run stores a hash of its predictions, and `repro` compares it on a re-run.
- Numbers in generated CSV files use fixed decimals so the same seed gives the same bytes.

## Consequences
- Re-running an identical config on identical data gives an identical predictions hash on
  every platform tested (Linux, Windows) and Python version (3.11 to 3.13).
- Floating-point behaviour can still differ across platforms in principle, so `repro` reports
  environment drift instead of promising a match.
- `1` and `1.0` in a config hash differently.
