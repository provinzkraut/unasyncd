# Unasync

A tool to transform asynchronous Python code to synchronous Python code.

## Why?

Unasyncd is largely inspired by [unasync](https://github.com/python-trio/unasync), and
a detailed discussion about this approach can be found
[here](https://github.com/urllib3/urllib3/issues/1323).

Its purpose is to reduce to burden of having to maintain both a synchronous and an
asynchronous version of otherwise functionally identical code. The idea behind simply
"taking out the async" is that often, synchronous and asynchronous code only differ
slightly: A few `await`s, `async def`s, `async with`s, and a couple of different method
names. The unasync approach makes use of this by treating the asynchronous version as a
source of truth from wich the synchronous version is then generated.

## Why unasyncd?

The original [unasync](https://github.com/python-trio/unasync) works by simply replacing
certain token, which is enough for most basic use cases, but can be somewhat restrictive
in the way the code can be written. More complex cases such as exclusion of functions /
classes or transformations (such as `AsyncExitStack` to `ExitStack` wich have not only
different names but also different method names that then need to be replaced only
within a certain scope) are not possible. This can lead to the introduction of shims,
introducing additional complexity.

Unasyncd's goal is to impose as little restrictions as possible to the way the
asynchronous code can be written, as long as it maps to a functionally equivalent
synchronous version.

To achieve this, unasyncd leverages [libcst](https://libcst.readthedocs.io/), enabling a
more granular control and complex transformations.

Unasyncd features:

1. Transformation of arbitrary modules, not bound to any specific directory structure
2. (Per-file) Exclusion of (nested) functions, classes and methods
3. Optional transformation of docstrings
4. Replacements based on fully qualified names
   (e.g. `typing.AsyncGenerator` is different than `foo.typing.AsyncGenerator`)
5. Transformation of constructs like `asyncio.TaskGroup` to a thread based equivalent

*A full list of supported transformations is available below.*

## Table of contents

<!-- TOC -->
* [Unasync](#unasync)
  * [Why?](#why)
  * [Why unasyncd?](#why-unasyncd)
  * [Table of contents](#table-of-contents)
  * [What can be transformed?](#what-can-be-transformed)
    * [Asynchronous functions](#asynchronous-functions)
    * [`await`](#await)
    * [Asynchronous iterators, iterables and generators](#asynchronous-iterators-iterables-and-generators)
    * [Asynchronous iteration](#asynchronous-iteration)
    * [Asynchronous context managers](#asynchronous-context-managers)
    * [`contextlib.AsyncExitStack`](#contextlibasyncexitstack)
    * [`asyncio.TaskGroup`](#asynciotaskgroup)
    * [`anyio.create_task_group`](#anyiocreatetaskgroup)
    * [`asyncio.sleep` / `anyio.sleep`](#asynciosleep--anyiosleep)
    * [Type annotations](#type-annotations)
    * [Docstrings](#docstrings)
  * [Usage](#usage)
    * [Installation](#installation)
    * [CLI](#cli)
    * [As a pre-commit hook](#as-a-pre-commit-hook)
    * [Configuration](#configuration)
      * [File](#file)
      * [CLI options](#cli-options)
      * [Exclusions](#exclusions)
      * [Extending name replacements](#extending-name-replacements)
    * [Handling of imports](#handling-of-imports)
    * [Integration with linters](#integration-with-linters)
    * [Limitations](#limitations)
    * [Disclaimer](#disclaimer)
<!-- TOC -->

## What can be transformed?

Unasyncd supports a wide variety of transformation, ranging from simple name
replacements to more complex transformations such as task groups.

### Asynchronous functions

*Async*
```python
async def foo() -> str:
    return "hello"
```

*Sync*
```python
def foo() -> str:
    return "hello"
```

### `await`

*Async*
```python
await foo()
```

*Sync*
```python
foo()
```

### Asynchronous iterators, iterables and generators

*Async*
```python
from typing import AsyncGenerator

async def foo() -> AsyncGenerator[str, None]:
    yield "hello"
```

*Sync*
```python
from typing import Generator

def foo() -> Generator[str, None, None]:
    yield "hello"
```

*Async*
```python
from typing import AsyncIterator

class Foo:
    async def __aiter__(self) -> AsyncIterator[str]:
        ...

    async def __anext__(self) -> str:
        raise StopAsyncIteration
```

*Sync*
```python
from typing import Iterator

class Foo:
    def __next__(self) -> str:
        raise StopIteration

    def __iter__(self) -> Iterator[str]:
        ...
```

*Async*
```python
x = aiter(foo)
```

*Sync*
```python
x = iter(foo)
```

*Async*
```python
x = await anext(foo)
```

*Sync*
```python
x = next(foo)
```

### Asynchronous iteration

*Async*
```python
async for x in foo():
    pass
```

*Sync*
```python
for x in foo():
    pass
```

*Async*
```python
[x async for x in foo()]
```

*Sync*
```python
[x for x in foo()]
```

### Asynchronous context managers

*Async*
```python
async with foo() as something:
    pass
```

*Sync*
```python
with foo() as something:
    pass
```

*Async*
```python
class Foo:
    async def __aenter__(self):
        ...

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        ...
```

*Sync*
```python
class Foo:
    def __enter__(self):
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...
```

*Async*
```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator

@asynccontextmanager
async def foo() -> AsyncGenerator[str, None]:
    yield "hello"
```

*Sync*
```python
from contextlib import contextmanager
from typing import Generator

@contextmanager
def foo() -> Generator[str, None, None]:
    yield "hello"
```

### `contextlib.AsyncExitStack`

*Async*
```python
import contextlib

async with contextlib.AsyncExitStack() as exit_stack:
    exit_stack.enter_context(context_manager_one())
    exit_stack.push(callback_one)
    exit_stack.callback(on_exit_one)

    await exit_stack.enter_async_context(context_manager_two())
    exit_stack.push_async_exit(on_exit_two)
    exit_stack.push_async_callback(callback_two)

    await exit_stack.aclose()
```

*Sync*
```python
import contextlib

with contextlib.ExitStack() as exit_stack:
    exit_stack.enter_context(context_manager_one())
    exit_stack.push(callback_one)
    exit_stack.callback(on_exit_one)

    exit_stack.enter_context(context_manager_two())
    exit_stack.push(on_exit_two)
    exit_stack.callback(callback_two)

    exit_stack.close()
```

See [limitations](#limitations)

### `asyncio.TaskGroup`

*Async*
```python
import asyncio

async with asyncio.TaskGroup() as task_group:
    task_group.create_task(something(1, 2, 3, this="that"))
```

*Sync*
```python
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.submit(something, 1, 2, 3, this="that")
```

See [limitations](#limitations)


### `anyio.create_task_group`

*Async*
```python
import anyio

async with anyio.create_task_group() as task_group:
    task_group.start_soon(something, 1, 2, 3)
```

*Sync*
```python
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.submit(something, 1, 2, 3)
```

See [limitations](#limitations)

### `asyncio.sleep` / `anyio.sleep`

Calls to `asyncio.sleep` and `anyio.sleep` will be replaced with calls to `time.sleep`:

*Async*
```python
import asyncio

await asyncio.sleep(1)
```

*Sync*
```python
import time

time.sleep(1)
```

If the call argument is `0`, the call will be replaced entirely:

```python
import asyncio

await asyncio.sleep(0)
```

### Type annotations

|                                   |                                    |
|-----------------------------------|------------------------------------|
| `typing.AsyncIterable[int]`       | `typing.Iterable[int]`             |
| `typing.AsyncIterator[int]`       | `typing.Iterator[int]`             |
| `typing.AsyncGenerator[int, str]` | `typing.Generator[int, str, None]` |


### Docstrings

Simply token replacement is available in docstrings:

*Async*
```python
async def foo():
    """This calls ``await bar()`` and ``asyncio.sleep``"""
```

*Sync*
```python
def foo():
    """This calls ``bar()`` and ``time.sleep``"""
```


## Usage

### Installation

```shell
pip install unasyncd
```

### CLI

Invoking `unasyncd` without any parameters will apply the configuration from the config
file:

```shell
unasyncd
```

But it's also possible to specify the files to be transformed directly:

```shell
unasyncd async_thing.py:aync_thing.py
```

This will transform `async_thing.py` and write the result back into `sync_thing.py`

### As a pre-commit hook

Unasyncd is available as a pre-commit hook:

```yaml
- repo: https://github.com/provinzkraut/unasyncd
  rev: v0.2.1
  hooks:
    - id: unasyncd
```

### Configuration

Unasyncd can be configured via a `pyproject.toml` file, a dedicated `.unasyncd.toml`
file or the command line interface.

#### File

| config file key               | type  | default | description                                                                        |
|-------------------------------|-------|---------|------------------------------------------------------------------------------------|
| `files`                       | table | -       | A table mapping source file names / directories to target file names / directories |
| `exclude`                     | array | -       | An array of names to exclude from transformation                                   |
| `per_file_exclude`            | table | -       | A table mapping files names to an array of names to exclude from transformation    |
| `add_replacements`            | table | -       | A table of additional name replacements                                            |
| `per_file_add_replacements`   | table | -       | A table mapping file names to tables of additional replacements                    |
| `transform_docstrings`        | bool  | false   | Enable transformation of docstrings                                                |
| `add_editors_note`            | bool  | false   | Add a note on top of the generated files                                           |
| `infer_type_checking_imports` | bool  | true    | Infer if new imports should be added to an 'if TYPE_CHECKING' block                |
| `cache`                       | bool  | true    | Cache transformation results                                                       |
| `force_regen`                 | bool  | false   | Always regenerate files, regardless if their content has changed                   |
| `ruff_fix`                    | bool  | false   | Run `ruff --fix` on the generated code                                             |

**Example**

```toml
[tool.unasyncd]
files = { "async_thing.py" = "sync_thing.py", "foo.py" = "bar.py" }
exclude = ["Something", "SomethingElse.within"]
per_file_exclude = { "foo.py" = ["special_foo"] }
add_replacements = { "my_async_func" = "my_sync_func" }
per_file_add_replacements = { "async_thing.py" = { "AsyncClass" = "SyncClass" } }
transform_docstrings = true
remove_unused_imports = false
no_cache = false
no_cache = false
force_regen = false
```

#### CLI options

*Feature flags corresponding to configuration values. These will override the
configuration file values*

| option                             | description                                                         |
|------------------------------------|---------------------------------------------------------------------|
| `--cache`                          | Cache transformation results                                        |
| `--no-cache `                      | Don't cache transformation results                                  |
| `--transform-docstrings`           | Enable transformation of docstrings                                 |
| `--no-transform-docstrings`        | Inverse of `--transform-docstrings`                                 |
| `--infer-type-checking-imports`    | Infer if new imports should be added to an 'if TYPE_CHECKING' block |
| `--no-infer-type-checking-imports` | Inverse of `infer-type-checking-imports`                            |
| `--add-editors-note`               | Add a note on top of each generated file                            |
| `--no-add-editors-note`            | Inverse of `--add-editors-note`                                     |
| `--ruff-fix`                       | Run `ruff --fix` on the generated code                              |
| `--no-ruff-fix`                    | Inverse of `--ruff-fix`                                             |
| `--force`                          | Always regenerate files, regardless if their content has changed    |
| `--no-force`                       | Inverse of `--force`                                                |
| `--check`                          | Don't write changes back to files                                   |
| `--write`                          | Inverse of `--check`                                                |


*Additional CLI options*

| option      | description                          |
|-------------|--------------------------------------|
| `--config`  | Alternative configuration file       |
| `--verbose` | Increase verbosity of console output |
| `--quiet`   | Suppress all console output          |


#### Exclusions

It is possible to exclude specific functions classes and methods from the
transformation. This can be achieved by adding their fully qualified name
(relative to the transformed module) under the `exclude` key:

```toml
[tool.unasyncd]
exclude = ["Something", "SomethingElse.within"]
```

In this example, classes or functions with the name `Something`, and the `within`
method of the `SomethingElse` class will be skipped.

The same option is available on a per-file basis, under the `per_file_exclude` key:

```toml
[tool.unasyncd]
per_file_exclude."module.py" = ["Something", "SomethingElse.within"]
```

This sets the same exclusion rules as above, but only for the file `module.py`.

#### Extending name replacements

Additional name replacement rules can be defined by adding fully qualified names
(relative to the transformed module) and replacements under the `add_replacements` key:

```toml
[tool.unasyncd]
add_replacements = { "some_module.some_name" = "some_other_module.some_other_name" }
```

The same option is available on a per-file basis, under the `per_file_add_replacements`
key:

```toml
[tool.unasyncd]
per_file_add_replacements."module.py" = { "some_module.some_name" = "some_other_module.some_other_name" }
```


### Handling of imports

Unasyncd will add new imports when necessary and tries to be sensible about the way it
does. There are however no guarantees about import order or compatibility with e.g.
isort or black. It follows a few basic rules:

1. Relativity of imports should be kept intact, e.g. `typing.AsyncGenerator` will be
   replaced with `typing.Generator` and `from typing import AsyncGenerator` with
   `from typing import Generator`
2. Existing imports will be updated if possible, for instance `from time import time`
   would become `from time import time, sleep` if `sleep` has been added by unasyncd
   during the transformation
3. New imports are added before the first non-import block that's not a docstring or a
   comment

Unasyncd will not remove imports that have become unused as a result of the applied
transformations. This is because tracking of usages is a complex task and best left to
tools made specifically for this job like [ruff](https://beta.ruff.rs/docs) or
[autoflake](https://github.com/PyCQA/autoflake).


### Integration with linters

Using unasyncd in conjunction with linters offering autofixing behaviour can lead to an
edit-loop, where unasyncd generates a new file which the other tool then changes in a
non-AST-equivalent way - for example by removing an import that has become unused as a
result of the transformation applied by unasyncd -, in turn causing unasyncd to
regenerate the file the next time it is invoked, since the target file is no longer
AST-equivalent to what unasyncd thinks it should be.

To alleviate this, unasyncd offers a [ruff](https://beta.ruff.rs/docs) integration,
which can automatically run `ruff --fix` on the generated code before writing it back.
It will use the existing ruff configuration for this to ensure the fixes applied to
adhere to the rules used throughout the project.

If this option is used, the transformed code will never be altered by ruff, therefore
breaking the cycle.

This option can be enabled with the `ruff_fix = true` feature flag, or by using the
`--ruff-fix` CLI flag.

Usage of this option requires an installation of `ruff`. If not independently installed,
it can be installed as an extra of unasyncd: `pip install unasyncd[ruff]`.

**Why is only ruff supported?**

Ruff was chosen for its speed, having a negligible impact on the overall performance of
unasyncd, and because it can replace most of the common linters / tools with autofixing
capabilities, removing the need for separate integrations.


### Limitations

Transformations for `contextlib.AsyncContextManager`, `asyncio.TaskGroup` and
`anyio.create_task_group` only work when they're being called in a `with` statement
directly. This is due to the fact that unasyncd does not track assignments or support
type inference. Support for these usages might be added in a future version.


### Disclaimer

Unasyncd's output should not be blindly trusted. While it is unlikely that it will break
things the resulting code should always be tested. Unasyncd is not intended to be run at
build time, but integrated into a git workflow (e.g. with
[pre-commit](https://pre-commit.com/)).
