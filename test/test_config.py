from pathlib import Path

import msgspec
import pytest

from unasyncd.config import Config, load_config


@pytest.fixture
def config_from_file(tmp_path: Path, monkeypatch):
    path = tmp_path / "pyproject.toml"

    def _create_file(**kwargs) -> Config:
        path.write_bytes(msgspec.toml.encode({"tool": {"unasyncd": kwargs}}))
        monkeypatch.chdir(tmp_path)
        return load_config(path)

    return _create_file


def test_default_config(config_from_file):
    config = config_from_file()

    assert config.add_editors_note is False
    assert config.transform_docstrings is False
    assert config.cache is True
    assert config.check_only is False
    assert config.force_regen is False
    assert config.files == {}
    assert config.exclude == {}
    assert config.extra_replacements == {}
    assert config.infer_type_checking_imports is True
    assert config.ruff_fix is False


def test_config_override(config_from_file, tmp_path):
    tmp_path.joinpath("foo.py").touch()

    config = config_from_file(
        add_editors_note=True,
        transform_docstrings=True,
        cache=False,
        check_only=True,
        force_regen=True,
        files={"foo.py": "bar.py"},
        infer_type_checking_imports=False,
        ruff_fix=True,
    )

    assert config.add_editors_note is True
    assert config.transform_docstrings is True
    assert config.cache is False
    assert config.check_only is True
    assert config.force_regen is True
    assert config.files == {"foo.py": "bar.py"}
    assert config.infer_type_checking_imports is False
    assert config.ruff_fix is True


def test_file_directories(config_from_file, tmp_path):
    source_dir = tmp_path / "foo"
    sub_dir = source_dir / "bar"
    file_1 = source_dir / "one.py"
    file_2 = source_dir / "two.py"
    file_3 = sub_dir / "three.py"

    for file in [file_1, file_2, file_3]:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()

    config = config_from_file(files={"foo": "foo_sync"})

    assert config.files == {
        "foo/bar/three.py": "foo_sync/bar/three.py",
        "foo/one.py": "foo_sync/one.py",
        "foo/two.py": "foo_sync/two.py",
    }


def test_excludes(config_from_file, tmp_path):
    tmp_path.joinpath("this.py").touch()

    config = config_from_file(
        files={"this.py": "that.py"},
        exclude=["Some.thing"],
        per_file_exclude={
            "foo.py": ["Something.else"],
            "bar.py": ["this", "And.that"],
        },
    )
    assert config.exclude == {
        "this.py": ["Some.thing"],
        "foo.py": ["Some.thing", "Something.else"],
        "bar.py": ["Some.thing", "this", "And.that"],
    }


def test_add_replacements(config_from_file, tmp_path):
    tmp_path.joinpath("this.py").touch()
    tmp_path.joinpath("that.py").touch()

    config = config_from_file(
        files={"this.py": "that.py"},
        add_replacements={"async_foo": "sync_foo"},
        per_file_add_replacements={
            "foo.py": {"this": "that"},
            "bar/baz.py": {"something": "else", "_and": "more_"},
        },
    )

    assert config.extra_replacements == {
        "this.py": {"async_foo": "sync_foo"},
        "foo.py": {"async_foo": "sync_foo", "this": "that"},
        "bar/baz.py": {"async_foo": "sync_foo", "something": "else", "_and": "more_"},
    }


def test_config_key(config_from_file):
    config = config_from_file()
    assert config.key() == config.key()

    old_key = config.key()
    config.force_regen = True
    assert config.key() != old_key


def test_config_key_changes_with_version(config_from_file, monkeypatch):
    config = config_from_file()
    old_key = config.key()

    import unasyncd.config

    monkeypatch.setattr(unasyncd.config, "VERSION", "99.0.0")

    assert old_key != config.key()
