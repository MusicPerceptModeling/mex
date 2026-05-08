import yaml


from pathlib import Path


def update_readme_config_entry(config_name: str, data_dir: str, readme_path: Path) -> None:
    content = readme_path.read_text() if readme_path.exists() else ''
    lines = content.splitlines()

    if len(lines) >= 2 and lines[0].strip() == '---':
        try:
            end_idx = next(i for i in range(1, len(lines)) if lines[i].strip() == '---')
        except StopIteration:
            end_idx = -1
    else:
        end_idx = -1

    if end_idx != -1:
        frontmatter_text = '\n'.join(lines[1:end_idx])
        body_text = '\n'.join(lines[end_idx + 1:])
    else:
        frontmatter_text = ''
        body_text = content

    frontmatter = yaml.safe_load(frontmatter_text) or {}
    configs = frontmatter.get('configs') or []

    replaced = False
    updated_configs: list[dict[str, str]] = []
    for entry in configs:
        if not isinstance(entry, dict):
            continue
        if entry.get('config_name') == config_name:
            if not replaced:
                updated_configs.append({'config_name': config_name, 'data_dir': data_dir})
                replaced = True
        else:
            updated_configs.append(entry)

    if not replaced:
        updated_configs.append({'config_name': config_name, 'data_dir': data_dir})

    frontmatter['configs'] = updated_configs
    dumped = yaml.safe_dump(frontmatter, sort_keys=False).rstrip()

    parts = ['---', dumped, '---']
    if body_text.strip():
        parts.append(body_text.rstrip())
    readme_path.write_text('\n'.join(parts) + '\n')