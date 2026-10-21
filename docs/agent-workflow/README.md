# Task-based agent workflow

Use this workflow for multi-step investigations, complex fixes, or work that must
survive a session handoff. Small, self-contained edits do not need these files.
[AGENTS.md](../../AGENTS.md) remains the source of project-wide rules; these
records describe one task and do not grant additional permissions.

## Three records

Copy the templates into a task-specific working directory:

| Record | Purpose |
| --- | --- |
| [TASK.md](TASK.md) | The agreed objective, scope, constraints, acceptance checks, budget and stopping conditions. |
| [LOG.md](LOG.md) | Current state, attempts, evidence, decisions and the next action. |
| [REVIEW.md](REVIEW.md) | An independent assessment tied to the exact changes inspected. |

Use a unique task directory under the scratch location configured for your
environment. Keep machine-specific paths in personal configuration. Do not
commit routine scratch artifacts. When preparing a PR, include the useful findings
and reproducible checks; provide accessible evidence links or attachments for
reviewers rather than relying only on local paths. Never retain credentials.

## Agree, execute, review

1. **Agree on the task.** Populate TASK from the user's request. Resolve choices
   that change the outcome, architecture, compatibility, dependencies or external
   state before dependent work. Do not ask again about already authorized work.
   TASK is the single source for the current acceptance rules. Record any agreed
   change there with its reason and update affected verification scripts together.
   Update the current scope and acceptance text when approval changes the task
   phase; an appended approval must not contradict an obsolete read-only boundary.
   An agent must not relax a gate just because an attempt failed it.
2. **Work within the boundary.** Read TASK and the current-state section of LOG.
   Inspect the live checkout before acting. For each meaningful attempt, state a
   hypothesis, make a scoped change, run the appropriate checks and record the
   outcome. Reuse existing tests and benchmark runners. Separate observations
   from causal explanations and open questions.
3. **Keep the handoff current.** Update LOG after a meaningful result or before
   ending a session. Put the current version, accepted findings, unresolved work
   and next action at the top. Keep failed attempts and corrections below; mark
   superseded conclusions instead of silently deleting them. Include evidence
   paths and exact commands, not a transcript of every tool call.
4. **Review independently.** Use a separate review session when authorized; do
   not automatically launch agents. The reviewer reads TASK, the actual diff,
   relevant code/tests and evidence, rather than relying on the author's summary.
   Reproduce candidate defects when practical. Record review findings separately
   from implementation; an author self-check is not an independent review.
5. **Hand back the decision.** Use REVIEW to distinguish accepted findings,
   incomplete checks and a completed review with no blocking findings. Fixes go
   back through verification and review. A review does not authorize a commit,
   push, merge, deployment or other action outside the user's existing approval.

## Evidence and completion

- Identify the reviewed base and head commits. For uncommitted work, retain a
  snapshot of the complete diff and any relevant untracked files, with hashes.
  A HEAD hash alone does not identify a dirty working tree. Later edits require
  review of the delta and an updated record; old results do not cover new code.
- For a fix, record whether the regression check fails before and passes after.
  If this cannot be established, explain the limitation. Do not rewrite the test
  to match the implementation merely to obtain a pass.
- For performance work, fix workload, quality requirements, timing scope and
  baseline before measuring. Retain raw successes and failures. A runner's
  successful exit is not evidence of a speedup; apply the task's quality and
  measurement criteria. See the [existing comparison workflow](../../benchmarks/hotpaths/README.md#alternating-comparisons-and-retained-evidence).
- A review's verdict is **PASS**, **CHANGES_REQUIRED**, or **INCOMPLETE**. PASS
  means the specified review and required checks completed with no blocking
  findings; it is not a proof of defect absence. Missing evidence or an unfinished
  review is INCOMPLETE, not PASS. Optional checks may be omitted with reasons.
- Distinguish local verification, PR checks, post-merge behavior and remote
  configuration when the task involves CI or deployment. Record the commit, run
  links and observed results for each applicable stage. A queued or missing run
  is pending, not a successful skip; a workflow definition does not configure
  branch protection. State which stages remain unverified.
- Stop when the agreed work is complete, the budget is reached, or further work
  needs missing information or authorization. Record the reason and remaining
  work. An investigation may end with no change; a budget stop is not task success.

The templates are records for people and agents, not an automated merge gate.
No GitHub bot, model choice, or additional framework is required.
