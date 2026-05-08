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

    if (submodule_path / ".git").exists():
        local_revision = subprocess.check_output(
            ["git", "-C", str(submodule_path), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(submodule_path), "status", "--porcelain"], text=True
            ).strip()
        )
        return staged_revision, local_revision, dirty
    return staged_revision, None, None


def read_readme_entries(submodule_path: Path) -> list[tuple[str, str]]:
    readme = submodule_path / "README.md"
    if not readme.exists():
        raise ValueError(f"Could not find {readme}")

    lines = readme.read_text().splitlines()
    sep_idx = [i for i, line in enumerate(lines) if line.strip() == "---"]
    if len(sep_idx) < 2:
        raise ValueError("README.md missing YAML frontmatter delimited by ---")

    frontmatter = lines[sep_idx[0] + 1 : sep_idx[1]]
    entries: list[tuple[str, str]] = []
    current_config_name: str | None = None

    for line in frontmatter:
        stripped = line.strip()
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        if stripped.startswith("config_name:"):
            current_config_name = stripped.split(":", 1)[1].strip().strip("'\"")
        elif stripped.startswith("data_dir:"):
            data_dir = stripped.split(":", 1)[1].strip().strip("'\"")
            if current_config_name and data_dir:
                entries.append((current_config_name, data_dir))
                current_config_name = None

    if not entries:
        raise ValueError("No config_name/data_dir entries found in README frontmatter")
    return entries


def list_submodule_dataset_dirs(submodule_path: Path) -> set[str]:
    return {
        p.name
        for p in submodule_path.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    }


def main() -> int:
    config_revision = read_config_revision()
    staged_revision, local_revision, dirty = read_submodule_revision()

    readme_entries: list[tuple[str, str]] | None = None
    readme_data_dirs: set[str] | None = None
    local_dirs: set[str] | None = None
    missing_in_readme: set[str] | None = None
    mismatched_entry_names: list[tuple[str, str]] | None = None
    readme_error: str | None = None

    if local_revision is not None:
        try:
            submodule_path = ROOT / SUBMODULE
            readme_entries = read_readme_entries(submodule_path)
            readme_data_dirs = {data_dir for _, data_dir in readme_entries}
            local_dirs = list_submodule_dataset_dirs(submodule_path)
            missing_in_readme = local_dirs - readme_data_dirs
            mismatched_entry_names = [
                (config_name, data_dir)
                for config_name, data_dir in readme_entries
                if config_name != data_dir
            ]
        except Exception as exc:
            readme_error = str(exc)

    msg = (
        "hf_datasets_mex validation failed:\n"
        f"  config/config.yaml: {config_revision}\n"
        f"  staged:             {staged_revision}\n"
        f"  local HEAD:         {local_revision}\n"
        f"  dirty:              {dirty}\n"
        f"  readme entries:     {readme_entries}\n"
        f"  readme data_dir:    {sorted(readme_data_dirs) if readme_data_dirs is not None else None}\n"
        f"  local dirs:         {sorted(local_dirs) if local_dirs is not None else None}\n"
        f"  missing in README:  {sorted(missing_in_readme) if missing_in_readme is not None else None}\n"
        f"  name mismatches:    {mismatched_entry_names}\n"
        f"  readme error:       {readme_error}"
    )

    has_error = (
        config_revision != staged_revision
        or (local_revision is not None and (config_revision != local_revision or dirty))
        or (readme_error is not None)
        or (missing_in_readme is not None and len(missing_in_readme) > 0)
        or (mismatched_entry_names is not None and len(mismatched_entry_names) > 0)
    )

    if has_error:
        print(msg)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())