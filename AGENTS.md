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
- If you need a verified environment, use marsbrain instead of assuming the local machine has all dependencies installed. See `agents/marsbrain-computation.md`.
- Do NOT push to remote unless approved.
- Typically, do not commit the files in `agents/computation` for privacy.

## Reference Implementation

- Reference repo: `https://github.com/amazon-far/holosoma` (retargeting module at `src/holosoma_retargeting`)
- You may clone the reference repo into a local directory on marsbrain for reading purposes.
- Do **not** import any code from the cloned reference directory.
- Do **not** massively copy code from the reference repo.
- Before every significant code implementation, check whether existing functions or utilities are already available in this repo before writing new ones.

## Verified Baseline

- Verified remote test baseline and marsbrain setup details are in `agents/marsbrain-computation.md`.
- `agents/computation/` may be empty in this repo because some computation configuration is private.
