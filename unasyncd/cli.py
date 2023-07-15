from __future__ import annotations

import asyncio
import time
from pathlib import Path

import rich
import rich_click as click

from .config import Config, load_config
from .main import unasync_files


async def _run(*, config: Config, check_only: bool, verbose: bool) -> bool:
    console = rich.get_console()
    verbose_console = rich.console.Console()
    verbose_console.quiet = not verbose

    start = time.perf_counter()
    files_changed = 0
    files_unchanged = 0

    status = console.status("Processing")
    if not verbose:
        status.start()

    async for result in unasync_files(config=config):
        if not result.transformed:
            files_unchanged += 1
            continue

        files_changed += 1

        verbose_console.print(
            f"Transformed [yellow]{result.source}[/] > [green]{result.target}[/]"
        )

    status.stop()

    console.print(f"Finished in {round(time.perf_counter() - start, 2)} seconds")
    console.print(
        f"[green]{files_changed}[/] file{'s' if files_changed != 1 else ''} "
        f"{'would be ' if check_only else ''}transformed, "
        f"[blue]{files_unchanged}[/] {'would be ' if check_only else ''}"
        "left unchanged"
    )

    return files_changed > 0


@click.command()
@click.option(
    "--cache/--no-cache",
    is_flag=True,
    help="Cache transformation results",
    default=None,
)
@click.option(
    "--transform-docstrings/--no-transform-docstrings",
    is_flag=True,
    default=None,
    help="Transform module, class, method and function docstrings",
)
@click.option(
    "--infer-type-checking-imports/--no-infer-type-checking-imports",
    is_flag=True,
    default=None,
    help="Infer if new imports should be added to an 'if TYPE_CHECKING' block",
)
@click.option(
    "--add-editors-note/--no-add-editors-note",
    is_flag=True,
    default=None,
    help="Add an editors note to the generated files",
)
@click.option(
    "--ruff-fix/--no-ruff-fix",
    is_flag=True,
    default=None,
    help="Run 'ruff --fix' on the transformed output before writing it back",
)
@click.option(
    "--check/--write",
    "check_only",
    is_flag=True,
    default=None,
    help="Write changes back",
)
@click.option(
    "-c",
    "--config",
    "config_file",
    help="Configuration file. If not given defaults to pyproject.toml",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "-f/-F",
    "--force/--no-force",
    "force_regen",
    help="Force regeneration regardless of changes",
    is_flag=True,
    default=None,
)
@click.option(
    "-v", "--verbose", is_flag=True, help="Increase verbosity of console output"
)
@click.option("-q", "--quiet", is_flag=True, help="Suppress all console output")
@click.argument("files", nargs=-1)
def main(
    files: tuple[str, ...],
    cache: bool | None,
    transform_docstrings: bool | None,
    infer_type_checking_imports: bool | None,
    ruff_fix: bool | None,
    check_only: bool,
    add_editors_note: bool | None,
    config_file: Path | None,
    force_regen: bool | None,
    verbose: bool,
    quiet: bool,
) -> None:
    console = rich.get_console()
    console.quiet = quiet

    if config_file is None:
        for name in ["pyproject.toml", ".unasyncd.toml"]:
            path = Path(name)
            if path.exists():
                config_file = path

    if config_file:
        console.print(f"[cyan]Config file: [yellow][b]{config_file}")

    config = load_config(
        path=config_file,
        cache=cache,
        transform_docstrings=transform_docstrings,
        add_editors_note=add_editors_note,
        files=dict(f.split(":", 1) for f in files) if files else None,
        check_only=check_only,
        force_regen=force_regen,
        infer_type_checking_imports=infer_type_checking_imports,
        ruff_fix=ruff_fix,
    )

    if not config.files:
        console.print("[red]No files selected. Quitting")
    else:
        quit(
            int(
                asyncio.run(_run(config=config, check_only=check_only, verbose=verbose))
            )
        )
