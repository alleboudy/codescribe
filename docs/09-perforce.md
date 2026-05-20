# 09 — Perforce primer

If your codebase lives in Perforce (Helix Core), here's what you need to know to make sense of the RAG source-client code in [issue #5 §1](https://github.com/alleboudy/codescribe/issues/5).

## What Perforce is and why some teams still use it

Perforce (officially "Helix Core" since ~2015) is a **centralised version control system** — the opposite of Git's distributed model. There's one authoritative server; clients sync files to local workspaces and submit changes back to the server in atomic units called **changelists**.

Reasons large teams stay on Perforce:

- **Scale**: Perforce handles 100M+ file repos well; Git struggles past a few million.
- **Atomic submits across binaries**: shipping a binary asset (texture, model, video) alongside the code that references it; Perforce treats them as one commit. Git LFS approximates but is fragile.
- **Granular permissions**: per-file ACLs; common in regulated/secure environments.
- **Established tooling**: build systems, CI, code review (Swarm) all integrated.
- **Streams**: a more disciplined branching model than Git branches.

Common contexts: game studios (Unreal/Unity engines ship as Perforce repos), defence contractors, large enterprise IT, telecom, semis. Anywhere with massive monorepos and binary assets.

## Core primitives

### Depot

The top-level container of files on the server. Path syntax starts with `//`: `//depot/main/src/foo.cpp`. A Perforce server can have multiple depots (e.g., `//depot/`, `//assets/`, `//tools/`); most setups use just one called `//depot/`.

### Workspace / Client

A *workspace* is a local checkout of part of the depot. Each workspace has a **client spec** that defines:
- The local root directory on disk.
- A **view** mapping depot paths to local paths (e.g., `//depot/main/... //my-client/main/...`).
- Options (compression, line endings, etc.).

You can have multiple clients per user (one per project, one per branch, etc.). The `P4CLIENT` environment variable selects which client is active.

### Changelist (CL)

A **changelist** is the atomic unit of change. Identified by a numeric ID — `CL 12345`. Globally numbered across the server (not per-branch).

States:
- **Pending** — you've added files to the CL but haven't submitted yet (Perforce's "staging").
- **Submitted** — committed to the depot; immutable.
- **Shelved** — a pending CL whose contents are stored on the server but not yet submitted (Perforce's "stash"). Used for code-review handoff.

Each submitted CL has:
- A description (free-text).
- An author (P4 username).
- A submit time.
- A list of file revisions (each with action: add/edit/delete/branch/integrate).
- A diff per file.

### Revision

Every file at every CL has a revision number, written `#N`. So `//depot/main/src/foo.cpp#5` is the 5th revision of that file. `#head` means the latest revision.

### Branch / Stream

Two flavours:

- **Classic branches**: a copy of a subtree of the depot to another subtree. e.g., `//depot/main/...` branched to `//depot/release-2.0/...`. Integration (merge-back) is manual but visible.
- **Streams**: a more structured branching model. Each stream has a type (mainline / development / release) and an explicit relationship to other streams. Modern Perforce setups use streams; older ones use classic branches.

### Ticket

Authentication credential. `p4 login` issues a ticket valid for ~12 hours. Stored in `$P4TICKETS` (default `~/.p4tickets`). The ticket is what subprocess calls present to the server.

## The `p4` CLI

The canonical Perforce client. A few sub-commands the RAG indexer uses:

### `p4 info`

Shows the configured server, user, client. First thing to run when debugging:

```bash
$ p4 info
User name: alice
Client name: alice-mac
Client host: alice.example.com
Client root: /Users/alice/perforce-workspace
Server address: ssl:perforce.corp.example.com:1666
Server license: ...
```

### `p4 login`

Authenticates against the server, issues a ticket. Interactive (asks for password) unless you're configured with SSO.

```bash
$ p4 login
Enter password: ***
User alice logged in.
```

### `p4 changes`

Lists changelists matching a filter. The RAG indexer uses this to enumerate submitted CLs.

```bash
# Last 5 submitted CLs to a depot path
$ p4 changes -s submitted -m 5 //depot/main/...

# All CLs since CL number 12345
$ p4 changes -s submitted //depot/main/...@12346,#head

# In Python marshal format (for parsing)
$ p4 -G changes -s submitted -m 5 //depot/main/...
```

The `-G` flag emits Python `marshal` binary format (one dict per record). The RAG client uses this in [issue #5 §1](https://github.com/alleboudy/codescribe/issues/5).

### `p4 describe`

Shows the contents of a CL: description + file list + (optionally) diff.

```bash
# Short form: description + file list, no diff
$ p4 describe -s 12345

# With unified diff for each file:
$ p4 describe -du 12345

# In marshal format:
$ p4 -G describe -s 12345
```

The `-du` flag produces a unified diff (the same format as `git diff`). The RAG client uses this to capture diffs into the store.

### `p4 print`

Prints the content of a single file at a given revision.

```bash
$ p4 print -q //depot/main/src/foo.cpp#5
```

The `-q` suppresses the metadata header. We don't use this in the RAG indexer (we have diffs from `describe -du`) but it's useful for ad-hoc lookups.

### `p4 sync`

Syncs files from the depot to your workspace. **The RAG indexer does NOT use this** — we only need metadata and diffs, not file content.

## The `p4 -G` (marshal) format

Most `p4` commands accept `-G` (capital G) to emit Python `marshal` binary instead of human-readable text. Each command produces a stream of records, one per logical entity (one per CL, one per file, etc.). To parse in Python:

```python
import marshal, subprocess, io

result = subprocess.run(
    ["p4", "-G", "changes", "-m", "5", "//depot/main/..."],
    capture_output=True, check=True,
)
buf = io.BytesIO(result.stdout)
records = []
while True:
    try:
        records.append(marshal.load(buf))
    except EOFError:
        break
```

Each record is a dict with **byte-string keys** (`b"change"`, `b"user"`, etc.) and byte-string values. Decode them with `.decode("utf-8", "replace")` to get strings.

Why marshal and not JSON? Historical — marshal predates JSON's ubiquity. Newer Perforce versions support `-Mj` (JSON output) but it's not universal across versions. We stick with `-G` for portability.

## Time stamps

Perforce stores timestamps as **Unix epoch seconds** in the `time` field. Convert in Python:

```python
from datetime import datetime, timezone
ts = datetime.fromtimestamp(int(meta["time"]), tz=timezone.utc)
```

## Common pitfalls

- **`p4 describe -du` truncates large diffs.** Diffs larger than ~1 MB are cut off with a `... truncated` marker. Capture this signal; downstream chunkers must handle truncated diffs. (See [issue #5 §1](https://github.com/alleboudy/codescribe/issues/5).)
- **`P4CLIENT` is required for some commands, optional for others.** `p4 changes` against an explicit depot path works without a client; some other commands fail without one. Document your setup.
- **Tickets expire mid-run.** A long-running indexer that takes >12 hours will hit `P4-AUTH` errors. Wrap subprocess calls with a clear error message — never store the password.
- **`p4 sync` is fast but disk-intensive**; don't accidentally sync the whole depot from the RAG indexer.
- **File path encoding**: depot paths are UTF-8 in modern Perforce but older servers may have non-UTF-8 file names in some places. Decode with `errors="replace"`.

## Auth flavours

Apart from password auth there's also:

- **SSO**: enterprise Perforce setups often integrate with SAML/OIDC. `p4 login` then redirects to a browser or relies on a system auth helper.
- **Auth ticket**: long-lived for headless/CI use. Generated via `p4 login -p` (prints the ticket value); store in `~/.p4tickets`.
- **OAuth tickets** (rare; some custom setups).

For the RAG indexer's purposes, all of these end up as a ticket in `~/.p4tickets`. The client just sets `P4TICKETS` env var and the subprocess uses it. Documented in [issue #5 §1](https://github.com/alleboudy/codescribe/issues/5)'s `_subprocess_env()` helper.

## Triggers (post-submit hooks)

Perforce supports server-side triggers that fire on submit, login, etc. The RAG indexer **doesn't** use these — we run incremental sync as a nightly cron — but if you want near-real-time indexing, you could:

1. Set up a `change-commit` trigger that hits an HTTP endpoint after each submit.
2. Have the endpoint enqueue a "refresh CL N" job that runs in the background.

This is documented as an open question in [issue #4 §19](https://github.com/alleboudy/codescribe/issues/4).

## Things Git users miss

- **No local commits.** You can `p4 edit` files and not submit, but they don't form a local history graph the way Git does.
- **No rebasing.** Submitted CLs are immutable. To "fix" history you obliterate (privileged) or branch+integrate.
- **No `git log -p file`.** Equivalent is `p4 filelog -L //depot/path/file` for descriptions, plus per-rev `p4 print` for content.
- **No `git blame`.** The equivalent is `p4 annotate //depot/path/file` (shows the CL number per line).
- **No cherry-pick across branches.** The equivalent is `p4 integrate` (more rigid; resolved at branch level, not per-commit).

## Things Perforce users miss when forced to Git

- **No atomic binary commits across the repo.** Git LFS exists but isn't atomic with regular commits.
- **No per-file ACLs.** Git permissions are repo-level.
- **No shelves** that persist server-side without affecting history.

## Further reading

- Perforce documentation: https://www.perforce.com/manuals/p4guide/
- The `p4` command reference: https://www.perforce.com/manuals/cmdref/
- Streams documentation: https://www.perforce.com/manuals/p4guide/Content/P4Guide/chapter.streams.html
- A good Git-to-Perforce mental model translation: https://www.perforce.com/blog/vcs/perforce-vs-git-which-better-development-team
