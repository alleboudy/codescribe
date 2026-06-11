"""``repo-docs`` MCP server package.

Exposes the target repo's (``../your-repo``'s) documentation files —
``README.md``, ``Makefile``, ``docs/**/*.md``, ``docker-compose*.yml``,
``Dockerfile*``, top-level ``*.sh`` — as MCP tools the agent can list
and read directly. Companion to ``repo-grep`` (lexical pattern
search) and ``repo-rag`` (semantic over commits/issues/PRs): this
server gives the model **structured access to operational docs and
build scripts** without going through retrieval.
"""
