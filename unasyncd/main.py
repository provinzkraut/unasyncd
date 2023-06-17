from __future__ import annotations

import ast
import asyncio
import hashlib
import shutil
import textwrap
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import anyio
import anyio.to_process
import msgspec
import msgspec.json
from anyio import Path

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from .config import Config


def _hash_ast(source: str) -> str:
    return hashlib.sha1(
        ast.unparse(ast.parse(source, type_comments=True)).encode("utf-8")
    ).hexdigest()


@dataclass
class TransformationResult:
    source: Path
    target: Path
    transformed: bool


class Cache:
    def __init__(self) -> None:
        self.cache_dir = Path(".unasyncd_cache")
        self.cache_content_dir = self.cache_dir / "content"

    async def _init_cache_dir(self) -> None:
        await self.cache_content_dir.mkdir(exist_ok=True, parents=True)
        cachedir_tag = self.cache_dir / "CACHEDIR.TAG"
        gitignore_file = self.cache_dir / ".gitignore"
        if not await cachedir_tag.exists():
            await cachedir_tag.write_text(
                textwrap.dedent(
                    """Signature: 8a477f597d28d172789f06886806bc55
                    # This file is a cache directory tag created by unasyncd.
                    # For information about cache directory tags, see:
                    #	http://www.brynosaurus.com/cachedir/
                    """
                )
            )
        if not await gitignore_file.exists():
            await gitignore_file.write_text("*\n")

    async def set(self, key: str, data: str) -> None:
        await self._init_cache_dir()
        cache_path = self.cache_content_dir / key
        await cache_path.write_text(data)

    async def get(self, key: str) -> str | None:
        if path := await self.get_path(key):
            return await path.read_text()
        return None

    async def get_path(self, key: str) -> Path | None:
        cache_path = self.cache_content_dir / key
        if await cache_path.exists():
            return cache_path

        return None


class FileMeta(msgspec.Struct):
    signature: str
    content_hash: str


class File(Path):
    """
    Wrapper around an ``anyio.Path`` that retrieves and caches metadata and
    contents.
    """

    def __init__(self, path: str | Path):
        super().__init__(path)
        self._content: str | None = None
        self._signature: str | None = None
        self._content_hash: str | None = None

    async def get_content(self) -> str:
        if self._content is None:
            self._content = await self.read_text()
        return self._content

    async def get_signature(self) -> str:
        if self._signature is None:
            stat = await self.stat()
            self._signature = f"{stat.st_mtime}:{stat.st_size}:{stat.st_mode}"
        return self._signature

    async def get_content_hash(self) -> str:
        if self._content_hash is None:
            self._content_hash = _hash_ast(await self.get_content())
        return self._content_hash

    async def to_meta(self) -> FileMeta:
        return FileMeta(
            signature=await self.get_signature(),
            content_hash=await self.get_content_hash(),
        )

    async def meta_equal(self, other: Any) -> bool:
        if isinstance(other, FileMeta):
            return bool(
                await self.get_signature() == other.signature
                or await self.get_content_hash() == other.content_hash
            )

        return False


class Env:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.cache = Cache()
        self._meta: dict[str, FileMeta] | None = None
        self._executor = ProcessPoolExecutor()
        self._config_key = self.config.key()

    async def _config_changed(self) -> bool:
        """
        Return a ``bool`` indicating if the current configuration has changed since the
        last run, by comparing its has with the hash stored in the ``__config_key_``
        cache file
        """
        if key := await self.cache.get("__config_key__"):
            return key != self._config_key
        return True

    async def get_meta(self, file: File) -> FileMeta | None:
        """Return the :class:`FileMeta` for ``file`` from the cache. If no meta has been
        recorded, return ``None``
        """
        if self._meta is None:
            if meta_raw := await self.cache.get("meta.json"):
                self._meta = msgspec.json.decode(meta_raw, type=dict[str, FileMeta])
            else:
                self._meta = {}

        return self._meta.get(str(file))

    async def set_meta(self, file: File) -> None:
        """Set the :class:`FileMeta` for ``file``"""
        if self._meta is None:
            self._meta = {}

        self._meta[str(file)] = await file.to_meta()

    async def _cache_key_for(self, file: File) -> str:
        """Return the cache key for ``file``"""
        return hashlib.sha1(
            (await file.get_content_hash() + self._config_key).encode()
        ).hexdigest()

    async def transform(self, file: File) -> str:
        """
        Return the transformed contents of ``file`` according to the current
        configuration
        """
        from .transformers import TreeTransformer

        transformer = TreeTransformer(
            exclude=self.config.exclude.get(str(file), set()),
            transform_docstrings=self.config.transform_docstrings,
            extra_name_replacements=self.config.extra_replacements.get(str(file), {}),
            infer_type_checking_imports=self.config.infer_type_checking_imports,
            ruff_fix=self.config.ruff_fix,
        )

        content = await file.get_content()
        loop = asyncio.get_running_loop()
        transformed = await loop.run_in_executor(self._executor, transformer, content)
        if self.config.add_editors_note:
            note = (
                "# Do not edit this file directly. It has been autogenerated from\n"
                f"# {file}\n"
            )
            transformed = note + transformed

        return transformed

    async def files_diverged(self, source: File, target: File) -> bool:
        """Return a bool indicating if ``source`` or ``target`` have diverged from
        their expected state.

        The files are considered equal if since the last run:

        - ``source_file`` stats have not changed
        - ``source_file`` is AST-equivalent to its previous version
        - ``target_file`` does exist

        If the ``force_regen`` option is used, always return ``True``.
        """
        if self.config.force_regen:
            return True

        if not await source.meta_equal(await self.get_meta(source)):
            return True

        if not await target.exists():
            return True

        return False

    async def _restore_from_cache(self, source: File, target: File) -> bool:
        """If a cached transformation for ``source`` exists, copy it to ``target``.
        If ``force_regen`` is ``True``, do nothing.
        """
        if self.config.force_regen:
            return False

        cached_file = await self.cache.get_path(await self._cache_key_for(source))
        if not cached_file:
            return False

        await target.parent.mkdir(parents=True, exist_ok=True)
        await anyio.to_thread.run_sync(shutil.copyfile, cached_file, target)
        return True

    async def unasync_file(
        self,
        source_path: str | Path,
        target_path: str | Path,
    ) -> TransformationResult:
        """
        Transform ``source_path`` and write the result back to ``target_path``, if the
        files have diverged according to :meth:`files_diverged`
        """
        source_file = File(source_path)
        target_file = File(target_path)

        if not await self.files_diverged(source_file, target_file):
            return TransformationResult(
                source=source_file, target=target_file, transformed=False
            )

        if self.config.cache:
            if await self._restore_from_cache(source_file, target_file):
                return TransformationResult(
                    source=source_file, target=target_file, transformed=True
                )

        await self.set_meta(source_file)

        if source_content := (await source_file.get_content()):
            transformed_content = await self.transform(source_file)
        else:
            transformed_content = source_content

        transformation_result = TransformationResult(
            source=source_file, target=target_file, transformed=False
        )

        if not self.config.check_only:
            if not (
                await target_file.exists()
                and await target_file.get_content() == transformed_content
            ):
                await target_file.parent.mkdir(parents=True, exist_ok=True)
                await target_file.write_text(transformed_content)
                transformation_result.transformed = True

            await self.set_meta(target_file)

        if self.config.cache:
            await self.cache.set(
                await self._cache_key_for(source_file), transformed_content
            )

        return transformation_result

    async def unasync_files(self) -> AsyncGenerator[TransformationResult, None]:
        """
        Transform all files specified in the configuration and return an async generator
        producing :class:`TransformationResult`.
        """
        if await self._config_changed():
            self.config.force_regen = True

        fs = []
        for source, target in self.config.files.items():
            fs.append(self.unasync_file(source, target))

        for c in asyncio.as_completed(fs):
            yield await c

        if self._meta:
            await self.cache.set("meta.json", msgspec.json.encode(self._meta).decode())

        await self.cache.set("__config_key__", self._config_key)


def unasync_files(config: Config) -> AsyncGenerator[TransformationResult, None]:
    """
    Transform all files specified in the configuration and return an async generator
    producing :class:`TransformationResult`.
    """
    env = Env(config=config)
    return env.unasync_files()
