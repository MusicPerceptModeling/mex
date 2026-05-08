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


def read_submodule_revision() -> str:
    submodule_path = ROOT / SUBMODULE
    if not (submodule_path / ".git").exists():
        raise RuntimeError(f"Submodule {SUBMODULE} not initialized. Run: git submodule update --init --recursive")
    
    staged_revision = subprocess.check_output(["git", "rev-parse", f":{SUBMODULE}"], text=True).strip()
    local_revision = subprocess.check_output(["git", "-C", str(submodule_path), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(submodule_path), "status", "--porcelain"], text=True).strip())
    return staged_revision, local_revision, dirty


def main() -> int:
    config_revision = read_config_revision()
    staged_revision, local_revision, dirty = read_submodule_revision()
    if config_revision != staged_revision or config_revision != local_revision or dirty:
        print(
            "hf_datasets_mex revision mismatch or dirty:\n"
            f"  config/config.yaml: {config_revision}\n"
            f"  staged:             {staged_revision}\n"
            f"  local HEAD:         {local_revision}\n"
            f"  dirty:              {dirty}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())