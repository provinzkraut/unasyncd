from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
from typing import Any

import msgspec


class Config(msgspec.Struct):
    add_editors_note: bool
    transform_docstrings: bool
    cache: bool
    extra_replacements: dict[str, dict[str, str]]
    exclude: dict[str, list[str]]
    files: dict[str, str]
    force_regen: bool
    check_only: bool
    infer_type_checking_imports: bool
    ruff_fix: bool

    def key(self) -> str:
        return hashlib.sha1(msgspec.json.encode(self)).hexdigest()


def _collect_paths(file_names: dict[str, str]) -> dict[str, str]:
    files = {}
    for source_name, target_name in file_names.items():
        source_path = Path(source_name)
        target_path = Path(target_name)

        if not source_path.exists():
            raise FileNotFoundError(str(source_path))

        if source_path.is_dir():
            for file in source_path.rglob("*.py"):
                files[str(file)] = str(target_path / (file.relative_to(source_path)))
        else:
            files[str(source_path)] = str(target_path)

    return files


def load_config(path: Path | None, **defaults: Any) -> Config:
    raw_config = (
        msgspec.toml.decode(path.read_bytes()).get("tool", {}).get("unasyncd", {})
        if path
        else {}
    )

    raw_config.update({k: v for k, v in defaults.items() if v is not None})

    exclude = raw_config.get("exclude", [])
    per_file_exclude = raw_config.get("per_file_exclude", {})

    extra_replacements = raw_config.get("add_replacements", {})
    per_file_extra_replacements = raw_config.get("per_file_add_replacements", {})

    files = _collect_paths(raw_config.get("files", {}))

    return Config(
        files=files,
        extra_replacements={
            file: {**extra_replacements, **per_file_extra_replacements.get(file, {})}
            for file in itertools.chain(files, per_file_extra_replacements)
        },
        exclude={
            file: [*exclude, *per_file_exclude.get(file, [])]
            for file in itertools.chain(files, per_file_exclude)
        },
        add_editors_note=raw_config.get("add_editors_note", False),
        transform_docstrings=raw_config.get("transform_docstrings", False),
        cache=raw_config.get("cache", True),
        check_only=raw_config.get("check_only", False),
        force_regen=raw_config.get("force_regen", False),
        infer_type_checking_imports=raw_config.get("infer_type_checking_imports", True),
        ruff_fix=raw_config.get("ruff_fix", False),
    )
