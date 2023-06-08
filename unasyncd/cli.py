import asyncio
import time
from pathlib import Path

import click
import rich

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
@click.option("--no-cache", is_flag=True, help="Disable caching", default=None)
@click.option(
    "-d",
    "--transform-docstrings",
    is_flag=True,
    default=None,
    help="Transform module, class, method and function docstrings. CST mode only",
)
@click.option(
    "-i",
    "--remove-unused-imports",
    is_flag=True,
    default=None,
    help="Remove imports that are unused as a result of the transformation. CST mode "
    "only",
)
@click.option(
    "--add-editors-note",
    is_flag=True,
    default=None,
    help="Don't add an editors note to the generated files",
)
@click.option("--check", "check_only", is_flag=True, help="Don't write changes back")
@click.option(
    "-c",
    "--config",
    "config_file",
    help="Configuration file. If not given defaults to pyproject.toml",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "-f",
    "--force",
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
    no_cache: bool | None,
    transform_docstrings: bool | None,
    remove_unused_imports: bool | None,
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
        no_cache=no_cache,
        transform_docstrings=transform_docstrings,
        remove_unused_imports=remove_unused_imports,
        add_editors_note=add_editors_note,
        files=dict(f.split(":", 1) for f in files) if files else None,
        check_only=check_only,
        force_regen=force_regen,
    )

    if not config.files:
        console.print("[red]No files selected. Quitting")
    else:
        quit(
            int(
                asyncio.run(_run(config=config, check_only=check_only, verbose=verbose))
            )
        )
