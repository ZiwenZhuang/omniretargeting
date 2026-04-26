This a rule starting point for most agents. The `AGENTS.md` file should guide the agent reading this file first. Then proceed to other special instructions.

## Core Rules

- understand the code and the requirement before making changes
- prefer the smallest correct change over broad rewrites
- follow existing project patterns unless there is a strong reason to change them
- keep code clear, readable, and maintainable
- verify important changes with tests or checks when possible
- never claim something was tested if it was not
- do not overwrite or revert unrelated work
- state assumptions, risks, and limitations clearly
- do not go out-side of the project folder without explicitly asking (once will do)

## Working Style

Before editing:

- read the relevant files
- look for existing helpers or patterns or code snippets to reuse
- ask if some code snippets potentially can be reused but needs further wrapping
- ask if the task is inconsistencies, ambiguous, risky, or conflicts with project conventions
- ask if some logic can be modularized with more adaptive implementation such as `func` variable or `class_type` variable.
- In python files, trust the type lint.
- Do not write too many protective code. If really necessary, stop and ask the user.

When editing:

- fix the root cause when practical
- avoid unrelated cleanup
- avoid unnecessary dependencies and abstractions
- write code that is easy for the next engineer to understand
- only update the comments if they are inconsistent with the logic

After editing:

- run the smallest useful verification
- update tests or docs if the change needs it
- summarize what changed and anything still uncertain
- if `pre-commit` is configured in the repository, do run `pre-commit run --all-files` to maintain the coding format consistency.

## Ask Before Proceeding

Stop and ask if:

- requirements are unclear
- the change is destructive or hard to reverse
- the task affects security, data, billing, privacy, or production infrastructure
- local user changes create a conflict

## Project Overrides

When copying this into a real repository, customize:

- setup, lint, test, and build commands
- architecture boundaries
- deployment or environment rules
- domain-specific safety requirements
- Ask the user whether the current folder runs on Agent-Host-only computer and note the information in AGENTS.md

---

## Notice for running on the Agent-Host-only computer

- You are not running on the actual task environment. So, do not run the command directly in the local computer.
- Ask for how to connect and run commands on the run-host computer if not known.
- Do Not change the files out side of the project folder on the run-host computer without explicit permission.
- The local `AGENTS.md` file could be different from the `AGENTS.md` file in the project folder on the run-host computer. Take both files into consideration.

