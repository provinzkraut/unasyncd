import secrets
import shutil
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from pytest import MonkeyPatch
from pytest_mock import MockerFixture

from unasyncd.cli import main
from unasyncd.main import Env


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    shutil.rmtree(".unasyncd_cache", ignore_errors=True)
    yield
    shutil.rmtree(".unasyncd_cache", ignore_errors=True)


TEST_CONTENT = textwrap.dedent(
    """
async def foo() -> None:
    pass
"""
)

TEST_TRANSFORMED_CONTENT = textwrap.dedent(
    """
def foo() -> None:
    pass
"""
)


@pytest.fixture()
def mock_cache(mocker: MockerFixture):
    return mocker.patch("unasyncd.main.Cache")


@pytest.fixture()
def source_file(tmp_path) -> Path:
    path = (tmp_path / secrets.token_hex()).with_suffix(".py")
    path.write_text(TEST_CONTENT)
    return path


@pytest.fixture()
def target_file(tmp_path) -> Path:
    return tmp_path.joinpath(secrets.token_hex()).with_suffix(".py")


@pytest.fixture()
def mock_transform(mocker: MockerFixture) -> MagicMock:
    return mocker.spy(Env, "transform")


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def mock_unasync_files(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("unasyncd.cli.unasync_files")


@pytest.fixture
def mock_load_config(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("unasyncd.cli.load_config")


@pytest.fixture()
def mock_string_transformer(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("unasyncd.main.StringTransformer").return_value


@pytest.fixture()
def mock_tree_transformer(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("unasyncd.tree_transformer.TreeTransformer").return_value


@pytest.mark.parametrize("config_file", ["pyproject.toml", ".unasyncd.toml"])
@pytest.mark.usefixtures("mock_unasync_files")
def test_default_config_file(
    runner: CliRunner,
    tmp_path: Path,
    mock_load_config: MagicMock,
    config_file: str,
    monkeypatch: MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / config_file
    config_path.touch()

    result = runner.invoke(main)

    assert not result.exception
    mock_load_config.assert_called_once()
    assert mock_load_config.call_args_list[0].kwargs["path"].absolute() == config_path


@pytest.mark.usefixtures("mock_unasync_files")
def test_set_config_file(
    runner: CliRunner, tmp_path: Path, mock_load_config: MagicMock
):
    config_path = tmp_path / "pyproject.toml"

    result = runner.invoke(main, ["-c", str(config_path)])

    assert not result.exception
    mock_load_config.assert_called_once()
    assert mock_load_config.call_args_list[0].kwargs["path"] == config_path


@pytest.mark.parametrize("no_infer_type_checking_import", [True, False])
@pytest.mark.parametrize("transform_docstrings", [True, False])
@pytest.mark.parametrize("add_editors_note", [True, False])
@pytest.mark.parametrize("check_only", [True, False])
@pytest.mark.parametrize("force_regen", [True, False])
def test_config_options(
    runner: CliRunner,
    mock_unasync_files: MagicMock,
    transform_docstrings: bool,
    add_editors_note: bool,
    check_only: bool,
    force_regen: bool,
    source_file: Path,
    target_file: Path,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    no_infer_type_checking_import: bool,
):
    monkeypatch.chdir(tmp_path)
    arguments = [f"{source_file}:{target_file}"]

    if transform_docstrings:
        arguments.append("--transform-docstrings")

    if add_editors_note:
        arguments.append("--add-editors-note")

    if check_only:
        arguments.append("--check")

    if force_regen:
        arguments.append("--force")

    if no_infer_type_checking_import:
        arguments.append("--no-infer-type-checking-imports")

    result = runner.invoke(main, arguments)

    assert not result.exception
    assert result.exit_code == 0
    config = mock_unasync_files.call_args_list[0].kwargs["config"]

    assert config.transform_docstrings is transform_docstrings
    assert config.add_editors_note is add_editors_note
    assert config.check_only is check_only
    assert config.force_regen is force_regen
    assert config.infer_type_checking_imports is not no_infer_type_checking_import


def test_pass_files(
    runner: CliRunner,
    mock_unasync_files: MagicMock,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    tmp_path.joinpath("foo.py").touch()
    tmp_path.joinpath("bar.py").touch()

    result = runner.invoke(main, ["foo.py:foo_sync.py", "bar.py:bar_sync.py"])

    assert not result.exception
    config = mock_unasync_files.call_args_list[0].kwargs["config"]

    assert config.files == {"foo.py": "foo_sync.py", "bar.py": "bar_sync.py"}


@pytest.mark.usefixtures("mock_unasync_files")
@pytest.mark.parametrize("files_transformed", [True, False])
def test_return_code(
    runner: CliRunner,
    mocker: MockerFixture,
    files_transformed: bool,
    source_file: Path,
    target_file: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.chdir(source_file.parent)
    mock_run = mocker.patch("unasyncd.cli._run")
    mock_run.return_value = files_transformed

    result = runner.invoke(main, f"{source_file}:{target_file}")
    assert result.exit_code == int(files_transformed)


def test_transform(runner: CliRunner, source_file: Path, target_file: Path) -> None:
    result = runner.invoke(main, [f"{source_file}:{target_file}"])

    assert result.exit_code == 1
    assert target_file.read_text() == TEST_TRANSFORMED_CONTENT


def test_transform_source_and_target_unchanged(
    runner: CliRunner, source_file: Path, target_file: Path, mock_transform: MagicMock
) -> None:
    args = [
        f"{source_file}:{target_file}",
    ]
    result_1 = runner.invoke(main, args)
    result_2 = runner.invoke(main, args)

    assert result_1.exit_code == 1
    assert result_2.exit_code == 0
    assert mock_transform.call_count == 1
    assert target_file.read_text() == TEST_TRANSFORMED_CONTENT


def test_transform_regen_config_changed(
    runner: CliRunner, source_file: Path, target_file: Path, mock_transform: MagicMock
) -> None:
    args = [f"{source_file}:{target_file}"]
    source_file.write_text(
        textwrap.dedent(
            """
    async def foo():
        '''this is a docstring with await bar()'''
    """
        )
    )

    expected_content = textwrap.dedent(
        """
    def foo():
        '''this is a docstring with bar()'''
    """
    )

    target_file.write_text(expected_content)

    result_1 = runner.invoke(main, args)
    result_2 = runner.invoke(main, [*args, "--transform-docstrings"])

    assert result_1.exit_code == 1
    assert result_2.exit_code == 1
    assert mock_transform.call_count == 2
    assert target_file.read_text() == expected_content


def test_transform_source_and_target_unchanged_force_regen(
    runner: CliRunner, source_file: Path, target_file: Path, mock_transform: MagicMock
) -> None:
    args = [f"{source_file}:{target_file}"]
    result_1 = runner.invoke(main, args)
    assert target_file.read_text() == TEST_TRANSFORMED_CONTENT
    assert result_1.exit_code == 1

    result_2 = runner.invoke(main, [*args, "--force"])

    assert result_2.exit_code == 0
    assert mock_transform.call_count == 2
    assert target_file.read_text() == TEST_TRANSFORMED_CONTENT


def test_transform_cached(
    runner: CliRunner, source_file: Path, target_file: Path, mock_transform: MagicMock
) -> None:
    args = [f"{source_file}:{target_file}"]
    result_1 = runner.invoke(main, args)
    target_file.unlink()

    result_2 = runner.invoke(main, args)

    assert result_1.exit_code == 1
    assert result_2.exit_code == 1
    assert mock_transform.call_count == 1
    assert target_file.read_text() == TEST_TRANSFORMED_CONTENT


def test_transform_cached_no_cache(
    runner: CliRunner, source_file: Path, target_file: Path, mock_transform: MagicMock
) -> None:
    args = [f"{source_file}:{target_file}", "--no-cache"]

    result_1 = runner.invoke(main, args)

    target_file.unlink()

    result_2 = runner.invoke(main, args)

    assert result_1.exit_code == 1
    assert result_2.exit_code == 1
    assert mock_transform.call_count == 2
    assert target_file.read_text() == TEST_TRANSFORMED_CONTENT


def test_transform_cached_force_regen(
    runner: CliRunner, source_file: Path, target_file: Path, mock_transform: MagicMock
) -> None:
    args = [f"{source_file}:{target_file}", "--force"]

    result_1 = runner.invoke(main, args)

    target_file.unlink()

    result_2 = runner.invoke(main, args)

    assert result_1.exit_code == 1
    assert result_2.exit_code == 1
    assert mock_transform.call_count == 2
    assert target_file.read_text() == TEST_TRANSFORMED_CONTENT


def test_transform_source_changed_ast_stable(
    runner: CliRunner, source_file: Path, target_file: Path, mock_transform: MagicMock
) -> None:
    args = [
        f"{source_file}:{target_file}",
    ]
    result_1 = runner.invoke(main, args)

    source_file.write_text(
        textwrap.dedent(
            """
    async def   foo() -> None :
        pass
    """
        )
    )

    result_2 = runner.invoke(main, args)

    assert result_1.exit_code == 1
    assert result_2.exit_code == 0
    assert mock_transform.call_count == 1
    assert target_file.read_text() == TEST_TRANSFORMED_CONTENT


def test_transform_target_changed_ast_stable(
    runner: CliRunner, source_file: Path, target_file: Path, mock_transform: MagicMock
) -> None:
    args = [
        f"{source_file}:{target_file}",
    ]
    result_1 = runner.invoke(main, args)

    target_content = textwrap.dedent(
        """
    def   foo() -> None :
        pass
    """
    )
    target_file.write_text(target_content)

    result_2 = runner.invoke(main, args)

    assert result_1.exit_code == 1
    assert result_2.exit_code == 0
    assert mock_transform.call_count == 1
    assert target_file.read_text() == target_content


def test_transform_source_and_target_changed_ast_stable(
    runner: CliRunner, source_file: Path, target_file: Path, mock_transform: MagicMock
) -> None:
    args = [
        f"{source_file}:{target_file}",
    ]
    result_1 = runner.invoke(main, args)

    source_file.write_text(
        textwrap.dedent(
            """
    async def   foo() -> None :
        pass
    """
        )
    )

    target_content = textwrap.dedent(
        """
    def   foo() -> None :
        pass
    """
    )
    target_file.write_text(target_content)

    result_2 = runner.invoke(main, args)

    assert result_1.exit_code == 1
    assert result_2.exit_code == 0
    assert mock_transform.call_count == 1
    assert target_file.read_text() == target_content


def test_transform_source_and_target_changed_no_transformation(
    runner: CliRunner, source_file: Path, target_file: Path, mock_transform: MagicMock
) -> None:
    args = [f"{source_file}:{target_file}"]
    result_1 = runner.invoke(main, args)

    target_file.write_text(TEST_TRANSFORMED_CONTENT + "\n\nimport foo")
    source_file.write_text(TEST_CONTENT + "\n\nimport foo")

    result_2 = runner.invoke(main, args)

    assert result_1.exit_code == 1
    assert result_2.exit_code == 0
    assert mock_transform.call_count == 2


def test_transform_add_editors_note(
    runner: CliRunner, source_file: Path, target_file: Path
) -> None:
    result = runner.invoke(main, [f"{source_file}:{target_file}", "--add-editors-note"])

    assert result.exit_code == 1
    expected_content = (
        f"# Do not edit this file directly. It has been autogenerated "
        f"from\n# {source_file}\n{TEST_TRANSFORMED_CONTENT}"
    )
    assert target_file.read_text() == expected_content
