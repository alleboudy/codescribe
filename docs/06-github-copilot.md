# 06 — GitHub Copilot in detail

GitHub Copilot is Microsoft/GitHub's umbrella brand for several distinct AI-assisted coding products. Knowing which one issue [#2](https://github.com/alleboudy/codescribe/issues/2) is targeting (and which Copilot can/cannot do what) matters for the build-off.

## The Copilot product family (as of mid-2026)

| Product | Where it lives | What it does | Use in this stack |
|---|---|---|---|
| **Copilot Code Completion** | Inline in IDE | Tab-complete style suggestions as you type | Out of scope here |
| **Copilot Chat** | IDE sidebar (VS Code, JetBrains, Visual Studio) | Conversational coding agent | Could drive the fine-tune if BYOK preview is available — see below |
| **Copilot Workspace** | Web (github.com/copilot-workspace) | Spec → plan → multi-file implementation flow | Best fit for the build-off (issue #2's per-phase prompts) |
| **Copilot Coding Agent** | Assigned to GitHub issues via `@copilot` | Autonomous PR creation from an issue body | Primary target of issue #2 per-phase prompts |
| **Copilot CLI** (`gh copilot`) | Terminal | Shell-command suggestions | Not relevant here |

The two that matter for this repo:

1. **Coding Agent** — what you `@copilot`-mention on a GitHub issue. It reads the issue, plans, opens a draft PR, makes commits, responds to review comments. Issue [#2](https://github.com/alleboudy/codescribe/issues/2)'s per-phase issue bodies are designed for this audience.
2. **Chat (Agent mode)** — the VS Code Copilot Chat panel switched to "Agent" mode, where the model can edit files and run terminal commands across your workspace. Comparable to `claw`/`aider`/`continue.dev` but cloud-hosted.

## How Copilot Coding Agent works

Lifecycle of a Copilot-assigned issue:

1. You file an issue with a clear deliverable + acceptance criteria.
2. Add `@copilot` as an assignee (or use the "Code with Copilot" button on the issue).
3. GitHub spawns a sandboxed VM ("Copilot's environment") that clones the repo on a new branch.
4. The sandbox VM has: the repo, GitHub CLI auth, a configurable set of allowed network destinations, and various pre-installed tools (git, common compilers, package managers).
5. Copilot reads:
   - The issue body (instructions).
   - `.github/copilot-instructions.md` at the repo root (project-wide rules).
   - `AGENTS.md` files at any directory level (scope-specific rules).
   - The repo's existing code and tests.
6. Copilot plans, edits, runs tests, iterates until either (a) it considers the work done or (b) it hits a wall it can't resolve.
7. It opens a **draft PR** linked to the issue with its commits.
8. A human reviews. PR review comments become further instructions for Copilot to iterate on.
9. Eventually a human approves and merges, or rejects.

### What Copilot reads (the "context files")

Both `.github/copilot-instructions.md` and any `AGENTS.md` files are read on every task. They're how this stack injects the strictly-local rules and per-package anti-patterns. Without them, Copilot would happily import `wandb` or default-bind `0.0.0.0` — both forbidden by the AGENTS.md system in issue [#2](https://github.com/alleboudy/codescribe/issues/2) Phase 0.

This is the "subagent" pattern in Copilot's world: instead of dynamically spawning subagents per task, you encode each package's rules in its `AGENTS.md` and let Copilot read them on every task in that package. Persistent context without re-pasting.

## How Copilot Chat (agent mode) works

Different beast:

- Lives inside VS Code (and JetBrains; here we focus on VS Code).
- The model is whatever the user has selected in the model picker (Claude 3.5/3.7 Sonnet, Claude Opus 4 family, GPT-4.x, etc. — depends on Copilot tier).
- "Agent mode" enables tool use (file edits, terminal commands, codebase search).
- Optional MCP support: in some preview builds, you can register MCP servers via settings.json.
- Permission system: per-tool approval prompts you can configure to auto-approve.

### Bring Your Own Key / Bring Your Own Model — the honest status

This is the load-bearing question for whether issue [#2](https://github.com/alleboudy/codescribe/issues/2)'s Phase 5 ("can we use our fine-tuned local model with Copilot Chat?") works:

| Path | Status as of 2026-05 | Reality |
|---|---|---|
| Copilot Chat → Anthropic BYOK | First-party, stable | Useless here — needs cloud Anthropic key. |
| Copilot Chat → Google Gemini BYOK | First-party, stable | Useless here — cloud Gemini key. |
| Copilot Chat → arbitrary OpenAI-compat endpoint | **Preview/tier-dependent** | Sometimes works via `github.copilot.chat.byok.openai.baseUrl` setting. Surface has shifted between Copilot updates. |
| Copilot Workspace → custom model | Not supported | Fixed catalogue. |
| GitHub Models API → custom local model | Not supported | The GitHub Models API is for *hosted* models on GitHub's infra. |
| `/etc/hosts` redirect of `api.githubcopilot.com` to a localhost proxy | Unsupported, brittle | Possible but breaks every Copilot update. |

**Practical recommendation**: don't bet the project on Copilot Chat consuming your local model. Issue [#2](https://github.com/alleboudy/codescribe/issues/2) explicitly drops "Copilot as harness" as an architectural premise. Use Copilot Coding Agent to *build the stack* (it works fine — it doesn't need to consume your fine-tuned model); then drive the fine-tuned model via `claw`/`aider`/`continue.dev`.

## Cloud-coupling implications

Even if BYOK works perfectly, **Copilot Chat's agent loop runs on GitHub's infrastructure**. The model that decides *which tool to call when* — the planner — is GitHub-hosted. Your fine-tune influences token generation; it does NOT replace the planner.

This means:

- The contents of files Copilot reads, the diffs it proposes, the shell commands it runs — all of this passes through GitHub's servers.
- If your codebase is under NDA, **Copilot Chat is not strictly-local**, regardless of what model produces the final completion.
- For strictly-local work, use a local harness (`claw`, `aider`, `continue.dev`) where both the agent loop AND the model run on your machine.

This is exactly the contradiction issue [#2](https://github.com/alleboudy/codescribe/issues/2)'s Phase 5 calls out and why "Copilot as harness" was dropped.

## When Copilot Coding Agent is the right tool

Use it when:

- The work is **well-scoped** to one or a few packages.
- The acceptance criteria are **CI-testable** (red tests → it iterates).
- The repo has **good AGENTS.md / copilot-instructions.md** files (Copilot reads them and stays on the rails).
- You don't need the implementer to use a *specific* model — Copilot picks from its own catalogue.
- The codebase is **either public, or your org's Copilot policy permits private-repo agent work**.

Issue [#2](https://github.com/alleboudy/codescribe/issues/2) is built around this pattern: 5 well-scoped phase issues, CI-enforced acceptance, dense AGENTS.md files. File each phase as an `@copilot`-assigned issue, wait for the PR, review, merge, move on.

## When Copilot Coding Agent is the wrong tool

- The work needs **specific local hardware** (training on YOUR GPU, not Copilot's sandbox). Issue [#2](https://github.com/alleboudy/codescribe/issues/2) handles this by marking GPU steps as "operator-verified post-merge" — Copilot ships the code; you run the training.
- The work is **deeply cross-cutting** (touches >5 packages in one PR). Copilot does worse with sprawl.
- The work needs **secrets or credentials** the sandbox doesn't have. (Copilot's sandbox CAN be configured with org-level secrets, but it's a deliberate setup.)

## Configuring the Copilot sandbox

Repository-level Copilot settings let you:

- **Allowlist outbound network destinations.** For this stack, you need `huggingface.co`, `pypi.org`, `github.com` (for `gh` CLI auth + releases), `nvidia.com` (CUDA repo). Configure under the repo's "Settings → Copilot → Agents → Allowed network destinations".
- **Inject env vars / secrets.** For things like `GH_TOKEN` for the GitHub source client (issue [#4](https://github.com/alleboudy/codescribe/issues/4) §8) or a Bugzilla API key fixture.
- **Set time budget.** Long tasks can be capped.

Without proper allowlist configuration, the very first phase that touches HuggingFace Hub will fail in the Copilot sandbox.

## Tips for Copilot-friendly issue writing

(This is the *prompt-engineering* angle for the build-off.)

1. **Lead with the title in conventional-commits format**: `feat(scope): action verb + scope`. Copilot picks up this style.
2. **List deliverables as explicit file paths.** Not "create the parser" — "create `<project>/data/parser.py` exposing `parse(text: str) -> AST`".
3. **State acceptance as CI-testable assertions**: "all tests in `tests/data/` green; no `print(` in `<project>/data/`; etc.".
4. **Reference AGENTS.md as the source of rules**, not the issue body. Saves repetition and keeps the rules in one place.
5. **Surface conflict-resolution policy.** "If a hard rule conflicts with this issue, surface in the PR description; do not silently pick." Copilot will often silently pick when given the chance.
6. **Cite history for non-obvious rules.** Saying "no `device_map='auto'`" is less effective than saying "no `device_map='auto'` — closed issue #3 in this repo's history details the KeyError chain it triggers".

Issue [#2 §16](https://github.com/alleboudy/codescribe/issues/2) (scoring criteria) measures exactly these dimensions.

## What Copilot is good at vs not good at

**Copilot Coding Agent is reliably good at:**
- Scaffolding new packages with consistent structure.
- Writing tests when the spec says exactly what to test.
- Following AGENTS.md rules — it really does read them.
- Iterating on PR review comments.
- Renames, mechanical refactors, lint fixes.

**Copilot is unreliable at:**
- Subtle algorithmic decisions (it'll often pick the first-thing-that-works without exploring trade-offs).
- Multi-file architectural design (asks for one file; it writes the file).
- "Just do whatever makes sense" — ambiguity is poison.
- Things that need real hardware (it can't run your GPU).

Issue [#2](https://github.com/alleboudy/codescribe/issues/2)'s phase structure is designed around these strengths: each phase is one package, with explicit deliverables and CI-enforced acceptance.

## Further reading

- Copilot Coding Agent docs: https://docs.github.com/copilot/concepts/about-copilot-coding-agent
- Copilot Chat docs: https://docs.github.com/copilot/using-github-copilot/asking-github-copilot-questions-in-your-ide
- Copilot Workspace: https://githubnext.com/projects/copilot-workspace
- `.github/copilot-instructions.md` reference: https://docs.github.com/copilot/customizing-copilot/about-customizing-github-copilot-chat-responses
- AGENTS.md convention (cross-tool): https://agents.md/
