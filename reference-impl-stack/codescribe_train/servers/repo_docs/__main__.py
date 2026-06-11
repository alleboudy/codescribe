"""Entry point: ``python -m codescribe_train.servers.repo_docs``.

Mirrors ``repo_grep``'s entry point so the same launch convention
applies. The framing-bridge wrapper in ``.claw/settings.json`` /
``.claw-mcp.json`` invokes this module path.
"""

from codescribe_train.servers.repo_docs.server import mcp

if __name__ == "__main__":
    mcp.run()
