# rag.sources — package rules (Git + GitHub)

Two read-only clients: `git` CLI subprocess + GitHub via `gh` CLI / GraphQL.

## Hard rules

- **Read-only.** No `git push`, no `git commit`, no `gh issue create`, no `gh pr create`, no PATCH/POST/PUT/DELETE against GitHub. Static check `tests/rag/sources/test_no_writes.py` greps for these.
- **Auth via `gh auth`** — no plaintext tokens in source. The `gh` CLI handles the token. If running outside `gh`, expect `GH_TOKEN` env var; never log it.
- **Allowlisted egress.** Only `api.github.com`. `httpx`-based wrapper around `gh api` is forbidden — call `gh` as a subprocess for consistent auth.
- **Rate-limit.** GitHub REST limit is 5000/h with auth; GraphQL is point-cost-based (5000 points/h). Use a token-bucket capped at 5 rps (3000/h budget) to leave headroom.
- **Resumable.** State watermark: `last_seen_commit_sha` (use `git log -1 HEAD` at bootstrap end) and `last_seen_issue_updated_at` (max `updated_at` across pulled issues).

## Specifics — Git source

- Uses `subprocess.run(["git", "-C", str(repo_path), ...], check=True, capture_output=True, text=True, timeout=60)`.
- Commit enumeration: `git log --all --since=<iso> --pretty=format:%H|%aI|%aE|%aN|%s` (pipe-separator).
- Diff for one commit: `git show <sha> --pretty=format:%B --patch -m -U3` (-m to handle merges as multiple diffs).
- NO `git clone`; the repo is already checked out.

## Specifics — GitHub source

- `gh api repos/example-org/sample/issues?state=all&since=<iso>&per_page=100 --paginate --jq '...'` for issues.
- `gh api repos/example-org/sample/pulls?state=all&sort=updated&direction=desc&per_page=100 --paginate` for PRs.
- `gh pr diff <number>` (or REST `pulls/<n>` with Accept: application/vnd.github.v3.diff) for PR diffs.
- GraphQL for closing references (this is the gold pairing signal):

      query {
        repository(owner: "example-org", name: "sample") {
          pullRequests(states: [MERGED, CLOSED], first: 100, after: <cursor>) {
            pageInfo { hasNextPage endCursor }
            nodes {
              number
              closingIssuesReferences(first: 20) { nodes { number } }
            }
          }
        }
      }

  Pull this in pages of 100; persist edge-list to `issue_pr_links` with source="closingIssuesReferences" and confidence=1.0.

## Anti-patterns

- Do NOT scrape GitHub HTML. Always API.
- Do NOT poll the issues endpoint at full speed; respect the 5 rps cap.
- Do NOT fetch the same PR twice in one run; dedup by PR number.
- Do NOT include drafts (PRs with `draft: true`) in the high-confidence corpus.
