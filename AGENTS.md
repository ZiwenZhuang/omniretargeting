# Agent Notes

## Project Shape

- Read `agents/CommonAgentInstructions.md` as the coding starting point.
- Read `agents/PROGRESS.md` for current development status before coding.
- Read computation resources in `agents/computation/` before any verified compute or test workflow.
- Python package for humanoid motion retargeting.
- Main CLI entry point: `python -m omniretargeting.main`
- Core implementation: `omniretargeting/core.py`
- Robot-profile loader: `omniretargeting/robot_config.py`
- Tests: `tests/`

## Working Rules

- Expect a dirty worktree. Do not revert unrelated local changes.
- Prefer small, surgical patches because this repo is actively edited and synchronized between machines.
- Follow the core OmniRetargeting math when designing APIs or refactors. This repository is a user-friendly implementation for future large-scale motion retargeting; convenience abstractions must not obscure the algorithm's actual inputs: mapped source target positions, robot link points, environment samples, interaction-mesh/Laplacian relationships, and explicit solver/post-processing configuration.
- Do not promote source-specific structures such as SMPL-X/BVH skeleton hierarchy, body-model internals, or per-source diagnostics into generic core contracts unless the retargeting algorithm directly consumes them. Keep those details in adapters, visualization helpers, or metadata.
- If you need a verified environment, use other resources instead of assuming the local machine has all dependencies installed. See available resources in `agents/computation/`.
- Do NOT push to remote unless approved.

## Reference Implementation

- Reference repo: `https://github.com/amazon-far/holosoma` (retargeting module at `src/holosoma_retargeting`)
- Reference paper: `https://arxiv.org/abs/2509.26633`
- Do **not** import any code from the cloned reference directory.
- Do **not** massively copy code from the reference repo.
- Before every significant code implementation, check whether existing functions or utilities are already available in this repo before writing new ones.

## Verified Baseline

- Verified remote test baseline and marsbrain setup details are in `agents/marsbrain-computation.md`.
- `agents/computation/` may be empty in this repo because some computation configuration is private.
