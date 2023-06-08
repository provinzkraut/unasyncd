# Unasync

A tool to transform asynchronous Python code to synchronous Python code.

## Why?

Unasyncd is largely inspired by [unasync](https://github.com/python-trio/unasync), and
a detailed discussion about this approach can be found
[here](https://github.com/urllib3/urllib3/issues/1323).

The goal is essentially to reduce to burden of having to maintain both a synchronous and
an asynchronous version of otherwise functionally identical code. The idea behind
simply "taking out the async" is that often, synchronous and asynchronous code only
differ very slightly: A few `await`s, `async def`s and `async with`s, and a couple of
different method names. The unasync approach makes use of this and provides a way to
use the asynchronous version as a source of truth from wich the synchronous version can
be generated.

## Why unasyncd?

The original [unasync](https://github.com/python-trio/unasync) takes a very simplistic
approach and works by replacing tokens. This works well enough for most basic use cases,
but can be somewhat restrictive in the way the code can be written. More complex cases
such as exclusion of functions / classes or transformations such as `AsyncExitStack` to
`ExitStack` are not possible, which leads to the introduction of shims, introducing
additional complexity.

Unasyncd leverages [libcst](https://libcst.readthedocs.io/), enabling a more granular
control and more complex transformations.

Unasyncd features:

1. Transformation of arbitrary modules, not bound to any specific directory structure
2. (Per-file) Exclusion of (nested) functions, classes and methods
3. Optional transformation of docstring
4. Replacements based on fully qualified names
   (e.g. `typing.AsyncGenerator` is different than `foo.typing.AsyncGenerator`)
5. Transformation of constructs like `asyncio.TaskGroup` to a thread based equivalent

*A full list of available transformations is available below.*

## Table of contents

* [Unasync](#unasync)
  * [Why?](#why)
  * [Why unasyncd?](#why-unasyncd)
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
      * [Options](#options)
      * [Exclusions](#exclusions)
      * [Extending name replacements](#extending-name-replacements)
    * [Handling of imports](#handling-of-imports)
    * [Integration with linters and formatters](#integration-with-linters-and-formatters)
    * [Limitations](#limitations)
    * [Disclaimer](#disclaimer)
<!-- TOC -->

## What can be transformed?

Unasyncd supports a wide variety of transformation, ranging from simple name
replacements to more complex transformations such as task groups.

### Asynchronous functions

Asynchronous functions and methods are replaced with a synchronous version:

```python
async def foo() -> str:
    return "hello"
```

```python
def foo() -> str:
    return "hello"
```

### `await`

`await` expressions will be unwrapped:

```python
await foo()
```

```python
foo()
```

### Asynchronous iterators, iterables and generators

```python
from typing import AsyncGenerator

async def foo() -> AsyncGenerator[str, None]:
    yield "hello"
```

```python
from typing import Generator

def foo() -> Generator[str, None, None]:
    yield "hello"
```

```python
from typing import AsyncIterator

class Foo:
    async def __aiter__(self) -> AsyncIterator[str]:
        ...

    async def __anext__(self) -> str:
        raise StopAsyncIteration
```

```python
from typing import Iterator

class Foo:
    def __next__(self) -> str:
        raise StopIteration

    def __iter__(self) -> Iterator[str]:
        ...
```

```python
x = aiter(foo)
```

```python
x = iter(foo)
```

```python
x = await anext(foo)
```

```python
x = next(foo)
```

### Asynchronous iteration

```python
async for x in foo():
    pass
```

```python
for x in foo():
    pass
```

### Asynchronous context managers

```python
async with foo() as something:
    pass
```

```python
with foo() as something:
    pass
```

```python
class Foo:
    async def __aenter__(self):
        ...

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        ...
```

```python
class Foo:
    def __enter__(self):
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...
```

```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator

@asynccontextmanager
async def foo() -> AsyncGenerator[str, None]:
    yield "hello"
```

```python
from contextlib import contextmanager
from typing import Generator

@contextmanager
def foo() -> Generator[str, None, None]:
    yield "hello"
```

### `contextlib.AsyncExitStack`

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

```python
import asyncio

async with asyncio.TaskGroup() as task_group:
    task_group.create_task(something(1, 2, 3, this="that"))
```

```python
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.submit(something, 1, 2, 3, this="that")
```

See [limitations](#limitations)


### `anyio.create_task_group`

```python
import anyio

async with anyio.create_task_group() as task_group:
    task_group.start_soon(something, 1, 2, 3)
```

```python
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.submit(something, 1, 2, 3)
```

See [limitations](#limitations)

### `asyncio.sleep` / `anyio.sleep`

Calls to `asyncio.sleep` and `anyio.sleep` will be replaced with calls to `time.sleep`:

```python
import asyncio

await asyncio.sleep(1)
```

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

```python
async def foo():
    """This calls ``await bar()`` and ``asyncio.sleep``"""
```

```python
async def foo():
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
  rev: v0.1.0
  hooks:
    - id: unasyncd
```

### Configuration

Unasyncd can be configured via a `pyproject.toml` file, a dedicated `.unasyncd.toml`
file or the command line interface.

#### Options

| config file key             | CLI                             | default | description                                                                        |
|-----------------------------|---------------------------------|---------|------------------------------------------------------------------------------------|
| `files`                     |                                 | -       | A table mapping source file names / directories to target file names / directories |
| `exclude`                   | N/A                             | -       | An array of names to exclude from transformation                                   |
| `per_file_exclude`          | N/A                             | -       | A table mapping files names to an array of names to exclude from transformation    |
| `add_replacements`          | N/A                             | -       | A table of additional name replacements                                            |
| `per_file_add_replacements` | N/A                             | -       | A table mapping file names to tables of additional replacements                    |
| `transform_docstrings`      | `-d` / `--transform-docstrings` | false   | Enable transformation of docstrings                                                |
| `add_editors_note`          | `--add-editors-note`            | false   | Add a note on top of the generated files                                           |
| `remove_unuse_imports`      | `--remove-unused-imports`       | false   | Remove imports that have become unused after the transformation                    |
| `no_cache`                  | `--no-cache`                    | false   | Disable caching                                                                    |
| `force_regen`               | `--force-regen`                 | false   | Always regenerate files, regardless if their content has changed                   |
| N/A                         | `-c` / `--config``              | -       | Specify an alternative configuration file                                          |

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

By default, unasyncd will not remove imports that have become unused as a result of the
applied transformations. This is because tracking of usages is a complex task and best
left to tools made specifically for this job like [ruff](https://beta.ruff.rs/docs) or
[autoflake](https://github.com/PyCQA/autoflake). It can also be a performance benefit
of not doing this work twice, e.g. when employing one of the aforementioned tools either
way.


### Integration with linters and formatters

Unasyncd should be run **after** tools that change the AST (e.g. isort, ruff), and
**before** tools that apply transformations in a way that does not change the AST
(e.g. black).

The reason for this is to avoid multiple passes being required until a stable state is
reached. Since unasyncd will only re-apply transformations to files which are no longer
AST equivalent, running it after tools that break AST equivalence solve this issue.

In practice this means all transformations aside from comments and formatting should be
applied before unasyncd.


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
