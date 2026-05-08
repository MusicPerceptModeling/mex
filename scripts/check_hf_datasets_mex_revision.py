from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "config.yaml"
SUBMODULE = "hf_datasets_mex"


def read_config_revision() -> str:
    for line in CONFIG.read_text().splitlines():
        if line.strip().startswith("hf_revision:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    raise ValueError(f"Could not find hf_revision in {CONFIG}")


def read_submodule_revision() -> tuple[str, str | None, bool | None]:
    submodule_path = ROOT / SUBMODULE
    staged_revision = subprocess.check_output(["git", "rev-parse", f":{SUBMODULE}"], text=True).strip()
    
    # Only check local state if submodule is initialized
    if (submodule_path / ".git").exists():
        local_revision = subprocess.check_output(["git", "-C", str(submodule_path), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(submodule_path), "status", "--porcelain"], text=True).strip())
        return staged_revision, local_revision, dirty
    else:
        return staged_revision, None, None


def main() -> int:
    config_revision = read_config_revision()
    staged_revision, local_revision, dirty = read_submodule_revision()
    
    # Build diagnostic message
    msg = (
        "hf_datasets_mex revision mismatch or dirty:\n"
        f"  config/config.yaml: {config_revision}\n"
        f"  staged:             {staged_revision}\n"
        f"  local HEAD:         {local_revision}\n"
        f"  dirty:              {dirty}"
    )
    
    # Check for failures
    has_error = (
        config_revision != staged_revision or 
        (local_revision is not None and (config_revision != local_revision or dirty))
    )
    
    if has_error:
        print(msg)
        return 1
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())