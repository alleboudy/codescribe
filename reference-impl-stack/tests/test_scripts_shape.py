"""Shape tests for the committed shell scripts.

These verify the committed shape (path, shebang, executable bit, strict mode)
so a typo at commit time surfaces in CI. We deliberately do NOT execute the
scripts — they build vendored binaries or launch long-running processes.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"

# Executable bash scripts shipped in the project. Sourced libraries
# (`_porter_lib.sh`) and the Windows PowerShell porter (`.ps1`) are excluded —
# they don't ship a bash shebang or +x bit by design.
BASH_SCRIPTS = (
    "bootstrap_linux.sh",
    "bootstrap_mac.sh",
    "build_claw_code.sh",
    "build_llama_cpp.sh",
    "relaunch-claw.sh",
    "run-claw.sh",
    "setup_wsl.sh",
)


@pytest.mark.parametrize("name", BASH_SCRIPTS)
def test_script_exists_and_is_executable(name: str) -> None:
    script = SCRIPTS / name
    assert script.is_file(), f"missing committed script: {script}"
    assert script.stat().st_mode & stat.S_IXUSR, (
        f"{script} is not user-executable (run: chmod +x {script})"
    )


@pytest.mark.parametrize("name", BASH_SCRIPTS)
def test_script_starts_with_bash_shebang(name: str) -> None:
    first_line = (SCRIPTS / name).read_text(encoding="utf-8").splitlines()[0]
    assert first_line == "#!/usr/bin/env bash", (
        f"{name} must start with `#!/usr/bin/env bash`; got {first_line!r}"
    )


@pytest.mark.parametrize("name", BASH_SCRIPTS)
def test_script_uses_strict_bash(name: str) -> None:
    body = (SCRIPTS / name).read_text(encoding="utf-8")
    assert "set -euo pipefail" in body, f"{name} is missing `set -euo pipefail`"
