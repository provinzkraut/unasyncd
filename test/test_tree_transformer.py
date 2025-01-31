from textwrap import dedent

import libcst as cst
import pytest
import tomli_w

from unasyncd.transformers import TreeTransformer


@pytest.fixture
def transformer() -> TreeTransformer:
    return TreeTransformer()


def test_unwrap_name_or_attribute() -> None:
    node = cst.Attribute(
        value=cst.Attribute(value=cst.Name("foo"), attr=cst.Name("bar")),
        attr=cst.Name("baz"),
    )
    assert cst.helpers.get_full_name_for_node_or_raise(node) == "foo.bar.baz"


def test_async_def(transformer: TreeTransformer) -> None:
    source = """
    async def foo():
        return None
    """

    expected = """
    def foo():
        return None
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_await(transformer: TreeTransformer) -> None:
    assert transformer("await foo") == "foo"
    assert transformer("await foo()") == "foo()"


def test_await_in_function(transformer: TreeTransformer) -> None:
    source = """
    async def foo():
        return await foo()
    """

    expected = """
    def foo():
        return foo()
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_async_for(transformer: TreeTransformer) -> None:
    source = """
    async for i in thing():
      pass
    """

    expected = """
    for i in thing():
      pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_async_for_with_await(transformer: TreeTransformer) -> None:
    source = """
    async for i in await thing():
      pass
    """

    expected = """
    for i in thing():
      pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_async_with(transformer: TreeTransformer) -> None:
    source = """
    async with foo() as bar:
      pass
    """

    expected = """
    with foo() as bar:
      pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_method(transformer: TreeTransformer) -> None:
    source = """
    class Foo:
        async def bar(self):
            pass
    """

    expected = """
    class Foo:
        def bar(self):
            pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_magic_method_replacement(transformer: TreeTransformer) -> None:
    source = """
    class Foo:
        async def __aenter__():
            pass

        async def __aexit__():
            pass

        async def __anext__():
            pass

        async def __aiter__():
            pass

        async def athrow():
            pass

        async def asend():
            pass
    """

    expected = """
    class Foo:
        def __enter__():
            pass

        def __exit__():
            pass

        def __next__():
            pass

        def __iter__():
            pass

        def throw():
            pass

        def send():
            pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_athrow(transformer: TreeTransformer) -> None:
    source = "await x.athrow()"
    expected = "x.throw()"

    assert transformer(source) == expected


def test_asend(transformer: TreeTransformer) -> None:
    source = "await x.asend()"
    expected = "x.send()"

    assert transformer(source) == expected


def test_anext(transformer: TreeTransformer) -> None:
    source = "anext(foo)"
    expected = "next(foo)"

    assert transformer(source) == expected


def test_aiter(transformer: TreeTransformer) -> None:
    source = "aiter(foo)"
    expected = "iter(foo)"

    assert transformer(source) == expected


def test_stop_async_iteration(transformer: TreeTransformer) -> None:
    source = """
    async def foo():
        yield
        raise StopAsyncIteration
    """

    expected = """
    def foo():
        yield
        raise StopIteration
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_asynccontextmanager(transformer: TreeTransformer) -> None:
    source = """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def foo():
        yield
    """

    expected = """
    from contextlib import asynccontextmanager, contextmanager

    @contextmanager
    def foo():
        yield
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_asynccontextmanager_alias_import(transformer: TreeTransformer) -> None:
    source = """
    from contextlib import asynccontextmanager as something_else

    @something_else
    async def foo():
        yield
    """

    expected = """
    from contextlib import asynccontextmanager as something_else, contextmanager

    @contextmanager
    def foo():
        yield
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_asynccontextmanager_not_from_contextlib_ignore(
    transformer: TreeTransformer,
) -> None:
    source = """
    async def asynccontextmanager():
        pass
    """

    expected = """
    def asynccontextmanager():
        pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_asynccontextmanager_module_import(transformer: TreeTransformer) -> None:
    source = """
    import contextlib

    @contextlib.asynccontextmanager
    async def foo():
        yield
    """

    expected = """
    import contextlib

    @contextlib.contextmanager
    def foo():
        yield
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_async_generator(transformer: TreeTransformer) -> None:
    source = """
    from typing import AsyncGenerator
    x: AsyncGenerator
    """

    expected = """
    from typing import AsyncGenerator, Generator
    x: Generator
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_async_generator_collections_abc(transformer: TreeTransformer) -> None:
    source = """
    from collections.abc import AsyncGenerator
    x: AsyncGenerator
    """

    expected = """
    from collections.abc import AsyncGenerator, Generator
    x: Generator
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_async_generator_annotation(transformer: TreeTransformer) -> None:
    source = """
    from typing import AsyncGenerator
    x: AsyncGenerator[str, int]
    """

    expected = """
    from typing import AsyncGenerator, Generator
    x: Generator[str, int, None]
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_async_generator_annotation_collections_abc(
    transformer: TreeTransformer,
) -> None:
    source = """
    from collections.abc import AsyncGenerator
    x: AsyncGenerator[str, int]
    """

    expected = """
    from collections.abc import AsyncGenerator, Generator
    x: Generator[str, int, None]
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_async_generator_annotation_typing_module_import(
    transformer: TreeTransformer,
) -> None:
    source = """
    import typing
    x: typing.AsyncGenerator[str, int]
    """

    expected = """
    import typing
    x: typing.Generator[str, int, None]
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_async_generator_class(transformer: TreeTransformer) -> None:
    source = """
    from typing import AsyncGenerator

    class Foo(AsyncGenerator[str, int]):
        pass
    """

    expected = """
    from typing import AsyncGenerator, Generator

    class Foo(Generator[str, int, None]):
        pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_async_generator_class_collections_abc(transformer: TreeTransformer) -> None:
    source = """
    from collections.abc import AsyncGenerator

    class Foo(AsyncGenerator[str, int]):
        pass
    """

    expected = """
    from collections.abc import AsyncGenerator, Generator

    class Foo(Generator[str, int, None]):
        pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_async_generator_class_typing_module_import(
    transformer: TreeTransformer,
) -> None:
    source = """
    import typing

    class Foo(typing.AsyncGenerator[str, int]):
        pass
    """

    expected = """
    import typing

    class Foo(typing.Generator[str, int, None]):
        pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_async_generator_class_collections_abc_module_import(
    transformer: TreeTransformer,
) -> None:
    source = """
    import collections.abc

    class Foo(collections.abc.AsyncGenerator[str, int]):
        pass
    """

    expected = """
    import collections.abc

    class Foo(collections.abc.Generator[str, int, None]):
        pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_async_iterable_annotation(transformer: TreeTransformer) -> None:
    source = """
    from typing import AsyncIterable
    x: AsyncIterable[str, int]
    """

    expected = """
    from typing import AsyncIterable, Iterable
    x: Iterable[str, int]
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_async_iterable_annotation_collections_abc(
    transformer: TreeTransformer,
) -> None:
    source = """
    from collections.abc import AsyncIterable
    x: AsyncIterable[str, int]
    """

    expected = """
    from collections.abc import AsyncIterable, Iterable
    x: Iterable[str, int]
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_async_iterable_annotation_typing_module_import(
    transformer: TreeTransformer,
) -> None:
    source = """
    import typing
    x: typing.AsyncIterable[str, int]
    """

    expected = """
    import typing
    x: typing.Iterable[str, int]
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_async_iterable_annotation_collections_abc_module_import(
    transformer: TreeTransformer,
) -> None:
    source = """
    import collections.abc
    x: collections.abc.AsyncIterable[str, int]
    """

    expected = """
    import collections.abc
    x: collections.abc.Iterable[str, int]
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_async_iterable_class(transformer: TreeTransformer) -> None:
    source = """
    from typing import AsyncIterable

    class Foo(AsyncIterable[str, int]):
        pass
    """

    expected = """
    from typing import AsyncIterable, Iterable

    class Foo(Iterable[str, int]):
        pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_async_iterable_class_collections_abc(transformer: TreeTransformer) -> None:
    source = """
    from collections.abc import AsyncIterable

    class Foo(AsyncIterable[str, int]):
        pass
    """

    expected = """
    from collections.abc import AsyncIterable, Iterable

    class Foo(Iterable[str, int]):
        pass
    """

    assert transformer(dedent(source)) == dedent(expected)  #


def test_async_iterable_class_typing_module_import(
    transformer: TreeTransformer,
) -> None:
    source = """
    import typing

    class Foo(typing.AsyncIterable[str, int]):
        pass
    """

    expected = """
    import typing

    class Foo(typing.Iterable[str, int]):
        pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_async_iterable_class_collections_abc_module_import(
    transformer: TreeTransformer,
) -> None:
    source = """
    import collections.abc

    class Foo(collections.abc.AsyncIterable[str, int]):
        pass
    """

    expected = """
    import collections.abc

    class Foo(collections.abc.Iterable[str, int]):
        pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_typing_import_not_stdlib(transformer: TreeTransformer) -> None:
    source = """
    from foo import typing

    x = typing.AsyncIterable
    """

    expected = """
    from foo import typing

    x = typing.AsyncIterable
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_async_iterator_annotation(transformer: TreeTransformer) -> None:
    source = """
    from typing import AsyncIterator
    x: AsyncIterator[str, int]
    """

    expected = """
    from typing import AsyncIterator, Iterator
    x: Iterator[str, int]
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_async_iterator_annotation_typing_module_import(
    transformer: TreeTransformer,
) -> None:
    source = """
    import typing
    x: typing.AsyncIterator[str, int]
    """

    expected = """
    import typing
    x: typing.Iterator[str, int]
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_async_iterator_class(transformer: TreeTransformer) -> None:
    source = """
    from typing import AsyncIterator

    class Foo(AsyncIterator[str, int]):
        pass
    """

    expected = """
    from typing import AsyncIterator, Iterator

    class Foo(Iterator[str, int]):
        pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_async_iterator_class_typing_module_import(
    transformer: TreeTransformer,
) -> None:
    source = """
    import typing

    class Foo(typing.AsyncIterator[str, int]):
        pass
    """

    expected = """
    import typing

    class Foo(typing.Iterator[str, int]):
        pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_async_exit_stack(transformer: TreeTransformer) -> None:
    source = """
    from contextlib import AsyncExitStack

    async with AsyncExitStack() as some_stack_name:
        await some_stack_name.enter_async_context(foo)
        some_stack_name.push_async_exit(bar)
        some_stack_name.push_async_callback(baz)
        await some_stack_name.aclose()
    """

    expected = """
    from contextlib import AsyncExitStack, ExitStack

    with ExitStack() as some_stack_name:
        some_stack_name.enter_context(foo)
        some_stack_name.push(bar)
        some_stack_name.callback(baz)
        some_stack_name.close()
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_async_exit_stack_contextlib_module_import(
    transformer: TreeTransformer,
) -> None:
    source = """
    import contextlib

    async with contextlib.AsyncExitStack() as some_stack_name:
        await some_stack_name.enter_async_context(foo)
        some_stack_name.push_async_exit(bar)
        some_stack_name.push_async_callback(baz)
        await some_stack_name.aclose()
    """

    expected = """
    import contextlib

    with contextlib.ExitStack() as some_stack_name:
        some_stack_name.enter_context(foo)
        some_stack_name.push(bar)
        some_stack_name.callback(baz)
        some_stack_name.close()
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_asyncio_task_group(transformer: TreeTransformer) -> None:
    source = """
    from asyncio import TaskGroup

    async with TaskGroup() as some_name:
        some_name.create_task(some_func())

        await something_else()

        async def nested_function():
            call()
            await async_call()
    """

    expected = """
    from asyncio import TaskGroup
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor() as some_name:
        some_name.submit(some_func)

        something_else()

        def nested_function():
            call()
            async_call()
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_asyncio_task_group_with_args(transformer: TreeTransformer) -> None:
    source = """
    from asyncio import TaskGroup

    async with TaskGroup() as some_name:
        some_name.create_task(some_func(1, foo=bar))
    """

    expected = """
    from asyncio import TaskGroup
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor() as some_name:
        some_name.submit(some_func, 1, foo=bar)
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_asyncio_task_alter_name(transformer: TreeTransformer) -> None:
    source = """
    from asyncio import TaskGroup

    async with TaskGroup() as task_group:
        task_group.create_task(some_func())

    def some_other_thing():
        task_group = None
    """

    expected = """
    from asyncio import TaskGroup
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor() as executor:
        executor.submit(some_func)

    def some_other_thing():
        task_group = None
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_asyncio_task_group_alias_import(transformer: TreeTransformer) -> None:
    source = """
    from asyncio import TaskGroup as Foobar

    async with Foobar() as task_group:
        task_group.create_task(some_func())
    """

    expected = """
    from asyncio import TaskGroup as Foobar
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor() as executor:
        executor.submit(some_func)
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_asyncio_task_group_module_import(transformer: TreeTransformer) -> None:
    source = """
    import asyncio

    async with asyncio.TaskGroup() as task_group:
        task_group.create_task(some_func())
    """

    expected = """
    import asyncio
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(some_func)
    """

    assert transformer(dedent(source)) == dedent(expected)


@pytest.mark.xfail
def test_asyncio_task_group_reassignment(transformer: TreeTransformer) -> None:
    source = """
    from asyncio import TaskGroup

    task_group = TaskGroup()

    async with task_group:
        task_group.create_task(some_func())
    """

    expected = """
    from concurrent.futures import ThreadPoolExecutor
    from asyncio import TaskGroup

    executor = ThreadPoolExecutor()

    with executor:
        executor.submit(some_func)
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_asyncio_task_missing_as_name_raises(transformer: TreeTransformer) -> None:
    source = """
    from asyncio import TaskGroup

    async with TaskGroup():
        pass
    """

    with pytest.raises(ValueError):
        transformer(dedent(source))


def test_task_group_not_from_asyncio_ignored(transformer: TreeTransformer) -> None:
    source = """
    class TaskGroup:
        pass

    async with TaskGroup():
        pass
    """

    expected = """
    class TaskGroup:
        pass

    with TaskGroup():
        pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_anyio_task_group(transformer: TreeTransformer) -> None:
    source = """
    from anyio import create_task_group

    async with create_task_group() as some_name:
        some_name.start_soon(some_func)
        some_name.start(some_other_func)

        await something_else()

        async def nested_function():
            call()
            await async_call()
    """

    expected = """
    from anyio import create_task_group
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor() as some_name:
        some_name.submit(some_func)
        some_name.submit(some_other_func)

        something_else()

        def nested_function():
            call()
            async_call()
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_anyio_task_group_with_args(transformer: TreeTransformer) -> None:
    source = """
    from anyio import create_task_group

    async with create_task_group() as task_group:
        task_group.start_soon(some_func, 1, 2)
    """

    expected = """
    from anyio import create_task_group
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor() as executor:
        executor.submit(some_func, 1, 2)
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_anyio_task_alter_name(transformer: TreeTransformer) -> None:
    source = """
    from anyio import create_task_group

    async with create_task_group() as task_group:
        task_group.start_soon(some_func)

    def some_other_thing():
        task_group = None
    """

    expected = """
    from anyio import create_task_group
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor() as executor:
        executor.submit(some_func)

    def some_other_thing():
        task_group = None
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_anyio_task_group_alias_import(transformer: TreeTransformer) -> None:
    source = """
    from anyio import create_task_group as something_else

    async with something_else() as task_group:
        task_group.start_soon(some_func)
    """

    expected = """
    from anyio import create_task_group as something_else
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor() as executor:
        executor.submit(some_func)
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_anyio_task_group_module_import(transformer: TreeTransformer) -> None:
    source = """
    import anyio

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(some_func)
    """

    expected = """
    import anyio
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(some_func)
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_anyio_task_missing_as_name_raises(transformer: TreeTransformer) -> None:
    source = """
    from anyio import create_task_group

    async with create_task_group():
        pass
    """

    with pytest.raises(ValueError):
        transformer(dedent(source))


def test_asyncio_sleep(transformer: TreeTransformer) -> None:
    source = """
    from asyncio import sleep
    await sleep(1)
    """

    expected = """
    from asyncio import sleep
    from time import sleep
    sleep(1)
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_asyncio_sleep_module_import(transformer: TreeTransformer) -> None:
    source = """
    import asyncio
    await asyncio.sleep(1)
    """

    expected = """
    import asyncio
    import time
    time.sleep(1)
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_asyncio_sleep_0(transformer: TreeTransformer):
    source = """
    import asyncio

    async def foo():
        await asyncio.sleep(0)
        return True
    """

    expected = """
    import asyncio

    def foo():
        return True
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_anyio_sleep(transformer: TreeTransformer) -> None:
    source = """
    from anyio import sleep
    await sleep(1)
    """

    expected = """
    from anyio import sleep
    from time import sleep
    sleep(1)
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_anyio_sleep_module_import(transformer: TreeTransformer) -> None:
    source = """
    import anyio
    await anyio.sleep(1)
    """

    expected = """
    import anyio
    import time
    time.sleep(1)
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_anyio_sleep_0(transformer: TreeTransformer):
    source = """
    import anyio

    async def foo():
        await anyio.sleep(0)
        return True
    """

    expected = """
    import anyio

    def foo():
        return True
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_relative_import(transformer: TreeTransformer) -> None:
    source = """
    from . import foo
    await foo()
    """

    expected = """
    from . import foo
    foo()
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_add_new_import_with_module_docstring() -> None:
    transformer = TreeTransformer()

    source = '''
    """This is a module level docstring"""
    import asyncio

    async def foo():
        await asyncio.sleep(1)
    '''

    expected = '''
    """This is a module level docstring"""
    import asyncio
    import time

    def foo():
        time.sleep(1)
    '''

    assert transformer(dedent(source)) == dedent(expected)


def test_add_new_import_with_module_docstring_multiline() -> None:
    transformer = TreeTransformer()

    source = '''
    """This is a module level docstring.
    It has multiple lines
    """
    import asyncio

    async def foo():
        await asyncio.sleep(1)
    '''

    expected = '''
    """This is a module level docstring.
    It has multiple lines
    """
    import asyncio
    import time

    def foo():
        time.sleep(1)
    '''

    assert transformer(dedent(source)) == dedent(expected)


def test_type_checking_import() -> None:
    transformer = TreeTransformer(
        extra_name_replacements={"foo.AsyncThing": "bar.SyncThing"},
    )

    source = """
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from foo import AsyncThing

    async def func() -> AsyncThing:
        ...
    """

    expected = """
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from foo import AsyncThing
        from bar import SyncThing

    def func() -> SyncThing:
        ...
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_reuse_existing_import() -> None:
    transformer = TreeTransformer(
        extra_name_replacements={"foo.AsyncThing": "bar.SyncThing"},
    )

    source = """
    from foo import AsyncThing
    from bar import SyncThing

    async def func() -> AsyncThing:
        ...
    """

    expected = """
    from foo import AsyncThing
    from bar import SyncThing

    def func() -> SyncThing:
        ...
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_add_to_existing_from_import() -> None:
    transformer = TreeTransformer(
        extra_name_replacements={"foo.async_func": "foo.sync_func"},
    )
    source = """
    from foo import bar, async_func

    async def call_func():
        await async_func()
    """

    expected = """
    from foo import bar, async_func, sync_func

    def call_func():
        sync_func()
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_unused_name_not_replaced() -> None:
    transformer = TreeTransformer()
    source = """
    from typing import AsyncIterable, Union

    x: AsyncIterable
    """

    expected = """
    from typing import AsyncIterable, Union, Iterable

    x: Iterable
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_unwrap_awaitable_annotation(transformer: TreeTransformer) -> None:
    source = """
    from typing import Awaitable, Callable
    from collections import abc


    async def foo() -> Awaitable[None]:
        pass

    async def bar(fn: Callable[[], Awaitable[int]]) -> None:
        pass

    async def baz() -> abc.Awaitable[None]:
        pass

    aw: Awaitable[list[dict[str, int]]]
    """

    expected = """
    from typing import Awaitable, Callable
    from collections import abc


    def foo() -> None:
        pass

    def bar(fn: Callable[[], int]) -> None:
        pass

    def baz() -> None:
        pass

    aw: list[dict[str, int]]
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_exclude_function() -> None:
    transformer = TreeTransformer(exclude=["foo"])

    source = """
    async def foo():
        pass

    async def bar():
        pass
    """

    expected = """
    async def foo():
        pass

    def bar():
        pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_exclude_function_nested() -> None:
    transformer = TreeTransformer(exclude=["foo.bar"])

    source = """
    async def foo():
        async def bar():
            pass
    """

    expected = """
    def foo():
        async def bar():
            pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_exclude_class() -> None:
    transformer = TreeTransformer(exclude=["Foo"])

    source = """
    class Foo:
        async def bar() -> None:
            pass

        async def baz() -> None:
            pass

    class Bar:
        async def foo() -> None:
            pass
    """

    expected = """
    class Foo:
        async def bar() -> None:
            pass

        async def baz() -> None:
            pass

    class Bar:
        def foo() -> None:
            pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_exclude_class_method() -> None:
    transformer = TreeTransformer(exclude=["Foo.baz"])

    source = """
    class Foo:
        async def bar() -> None:
            pass

        async def baz() -> None:
            pass
    """

    expected = """
    class Foo:
        def bar() -> None:
            pass

        async def baz() -> None:
            pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_exclude_nested_class() -> None:
    transformer = TreeTransformer(exclude=["Foo.Bar.baz"])

    source = """
    class Foo:
        class Bar:
            async def bar() -> None:
                pass

            async def baz() -> None:
                pass

        async def foo() -> None:
            pass
    """

    expected = """
    class Foo:
        class Bar:
            def bar() -> None:
                pass

            async def baz() -> None:
                pass

        def foo() -> None:
            pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_function_docstring(transformer: TreeTransformer) -> None:
    source = '''
    async def foo():
        """Some text with a ``await this()`` call.
        And here is another line.
        """
        pass
    '''

    expected = '''
    def foo():
        """Some text with a ``this()`` call.
        And here is another line.
        """
        pass
    '''

    assert transformer(dedent(source)) == dedent(expected)


def test_class_docstring(transformer: TreeTransformer) -> None:
    source = '''
    class Foo:
        """Some text with a ``await this()`` call.
        And here is another line.
        """
    '''

    expected = '''
    class Foo:
        """Some text with a ``this()`` call.
        And here is another line.
        """
    '''

    assert transformer(dedent(source)) == dedent(expected)


def test_module_docstring(transformer: TreeTransformer) -> None:
    source = '''
    """Some text with a ``await this()`` call.
    And here is another line.
    """
    '''

    expected = '''
    """Some text with a ``this()`` call.
    And here is another line.
    """
    '''

    assert transformer(dedent(source)) == dedent(expected)


def test_wrapped_await_expression_preserve_inline_comment(
    transformer: TreeTransformer,
) -> None:
    source = """
    foo = (  # type:ignore[no-any-return]  # pragma: no cover
        await something()
    )
    """

    expected = """
    foo = (  # type:ignore[no-any-return]  # pragma: no cover
        something()
    )
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_function_preserve_inline_comment(transformer: TreeTransformer) -> None:
    source = """
    async def foo(  # preserve this
    ) -> None:  # and this
        pass
    """

    expected = """
    def foo(  # preserve this
    ) -> None:  # and this
        pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_honour_type_checking_import() -> None:
    transformer = TreeTransformer(
        extra_name_replacements={
            "module_a.AsyncThing": "module_b.SyncThing",
            "async_module.Something": "sync_module.SomethingElse",
        },
    )

    source = """
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        import async_module
        from module_a import AsyncThing

    def bar(param: async_module.Something) -> AsyncThing:
        pass
    """

    expected = """
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        import async_module
        from module_a import AsyncThing
        from module_b import SyncThing
        import sync_module

    def bar(param: sync_module.SomethingElse) -> SyncThing:
        pass
    """
    assert transformer(dedent(source)) == dedent(expected)


def test_honour_type_checking_import_submodule() -> None:
    transformer = TreeTransformer(
        extra_name_replacements={
            "module_a.submodule_a.AsyncThing": "module_b.submodule_b.SyncThing",
            "module_c.sub_c.ThingC": "module_d.sub_d.ThingD",
        },
    )

    source = """
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from module_a.submodule_a import AsyncThing
        import module_c.sub_c

    def bar(param: module_c.sub_c.ThingC) -> AsyncThing:
        pass
    """

    expected = """
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from module_a.submodule_a import AsyncThing
        import module_c.sub_c
        from module_b.submodule_b import SyncThing
        import module_d.sub_d

    def bar(param: module_d.sub_d.ThingD) -> SyncThing:
        pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_honour_type_checking_import_multiple_names_from_same_module() -> None:
    transformer = TreeTransformer(
        extra_name_replacements={
            "foo.ThingOne": "bar.ThingThree",
            "foo.ThingTwo": "bar.ThingFour",
        },
    )

    source = """
    from typing import TYPE_CHECKING
    from foo import ThingOne

    if TYPE_CHECKING:
        from foo import ThingTwo

    def baz(param: ThingOne) -> ThingTwo:
        pass
    """

    expected = """
    from typing import TYPE_CHECKING
    from foo import ThingOne
    from bar import ThingThree

    if TYPE_CHECKING:
        from foo import ThingTwo
        from bar import ThingFour

    def baz(param: ThingThree) -> ThingFour:
        pass
    """
    result = transformer(dedent(source))
    assert result == dedent(expected)


def test_disable_infer_type_checking_imports() -> None:
    transformer = TreeTransformer(
        extra_name_replacements={
            "module_a.AsyncThing": "module_b.SyncThing",
        },
        infer_type_checking_imports=False,
    )

    source = """
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from module_a import AsyncThing

    def bar() -> AsyncThing:
        pass
    """

    expected = """
    from typing import TYPE_CHECKING
    from module_b import SyncThing

    if TYPE_CHECKING:
        from module_a import AsyncThing

    def bar() -> SyncThing:
        pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_preserve_import_formatting(transformer: TreeTransformer) -> None:
    source = """
    from typing import (
        AsyncGenerator,
        Generator,
    )
    """

    assert transformer(dedent(source)) == dedent(source)


def test_ruff_fix(tmp_path, monkeypatch) -> None:
    config_file = tmp_path / "ruff.toml"
    config_file.write_text(tomli_w.dumps({"lint": {"select": ["I001"]}}))
    monkeypatch.chdir(tmp_path)
    transformer = TreeTransformer(ruff_fix=True)

    source = """
    import time, asyncio
    """

    expected = """
    import asyncio
    import time
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_ruff_fix_ruff_error_raises(tmp_path, monkeypatch) -> None:
    config_file = tmp_path / "ruff.toml"
    config_file.write_text(tomli_w.dumps({"lint": {"select": ["FOO017"]}}))
    monkeypatch.chdir(tmp_path)
    transformer = TreeTransformer(ruff_fix=True)

    source = """
    import asyncio
    """

    with pytest.raises(ChildProcessError, match="Error calling ruff"):
        transformer(dedent(source))


def test_ruff_fix_pass_file_name(tmp_path, monkeypatch) -> None:
    config_file = tmp_path / "ruff.toml"
    config_file.write_text(
        tomli_w.dumps(
            {
                "lint": {
                    "per-file-ignores": {"some_file.py": ["I001"]},
                    "select": ["I001"],
                }
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    transformer = TreeTransformer(ruff_fix=True, file_name="some_file.py")

    source = """
    import time, asyncio
    """

    expected = """
    import time, asyncio
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_ruff_format(tmp_path, monkeypatch) -> None:
    config_file = tmp_path / "ruff.toml"
    config_file.write_text(tomli_w.dumps({"format": {"quote-style": "single"}}))
    monkeypatch.chdir(tmp_path)
    transformer = TreeTransformer(ruff_format=True)

    source = """mode = "format"
    """

    expected = """mode = 'format'
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_ruff_format_ruff_error_raises(tmp_path, monkeypatch):
    config_file = tmp_path / "ruff.toml"
    config_file.write_text(tomli_w.dumps({"format": {"quote-style": "quadruple"}}))
    monkeypatch.chdir(tmp_path)
    transformer = TreeTransformer(ruff_format=True)

    source = """
    mode = "format"
    """

    with pytest.raises(ChildProcessError, match="Error calling ruff"):
        transformer(dedent(source))


def test_ruff_fix_and_format(tmp_path, monkeypatch) -> None:
    config_file = tmp_path / "ruff.toml"
    config_file.write_text(
        tomli_w.dumps(
            {"format": {"quote-style": "single"}, "lint": {"select": ["I001"]}}
        )
    )
    monkeypatch.chdir(tmp_path)
    transformer = TreeTransformer(ruff_fix=True, ruff_format=True)

    source = """
    import time, asyncio
    mode = "format"
    """

    expected = """
    import asyncio
    import time

    mode = 'format'
    """

    assert transformer(dedent(source)) == dedent(expected).lstrip()


def test_async_comprehension(transformer: TreeTransformer) -> None:
    source = """
    [x async for x in foo()]
    {x async for x in foo()}
    (x async for x in foo())
    {x: 1 async for x in foo()}
    """

    expected = """
    [x for x in foo()]
    {x for x in foo()}
    (x for x in foo())
    {x: 1 for x in foo()}
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_replace_relative_imported_names() -> None:
    transformer = TreeTransformer(
        extra_name_replacements={".foo.AsyncThing": ".bar.SyncThing"}
    )
    source = """
    from .foo import AsyncThing

    async def func() -> AsyncThing:
        ...
    """

    expected = """
    from .foo import AsyncThing
    from .bar import SyncThing

    def func() -> SyncThing:
        ...
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_module_dunder_all_import():
    transformer = TreeTransformer(
        extra_name_replacements={
            ".foo.AsyncThing": ".foo.SyncThing",
            "bar.Something": "baz.SomethingElse",
        }
    )

    source = """
    from .foo import AsyncThing
    from bar import Something

    __all__ = ("AsyncThing", "Something")
    """

    expected = """
    from .foo import AsyncThing, SyncThing
    from bar import Something
    from baz import SomethingElse

    __all__ = ("SyncThing", "SomethingElse")
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_anyio_path(transformer):
    source = """
    from anyio import Path

    foo = await Path().read_text()
    """

    expected = """
    from anyio import Path
    from pathlib import Path

    foo = Path().read_text()
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_anyio_path_module_import(transformer):
    source = """
    import anyio

    foo = await anyio.Path().read_text()
    """

    expected = """
    import anyio
    import pathlib

    foo = pathlib.Path().read_text()
    """

    assert transformer(dedent(source)) == dedent(expected)


@pytest.mark.xfail(reason="not implemented")
def test_transform_imports_in_function(transformer):
    source = """
    def foo():
        from contextlib import asynccontextmanager
    """

    expected = """
    def foo():
        from contextlib import asynccontextmanager
        from contextlib import contextmanager
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_asyncio_semaphore(transformer):
    source = """
    import asyncio

    sem = asyncio.Semaphore(2)

    await sem.acquire()
    await sem.release()

    async with sem:
        pass
    """

    expected = """
    import asyncio
    import threading

    sem = threading.Semaphore(2)

    sem.acquire()
    sem.release()

    with sem:
        pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_anyio_semaphore(transformer):
    source = """
    import anyio

    sem = anyio.Semaphore(2)

    await sem.acquire()
    await sem.release()

    async with sem:
        pass
    """

    expected = """
    import anyio
    import threading

    sem = threading.Semaphore(2)

    sem.acquire()
    sem.release()

    with sem:
        pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_asyncio_lock(transformer):
    source = """
    import asyncio

    lock = asyncio.Lock()

    async with lock:
        pass
    """

    expected = """
    import asyncio
    import threading

    lock = threading.Lock()

    with lock:
        pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_anyio_lock(transformer):
    source = """
    import anyio

    lock = anyio.Lock()

    async with lock:
        pass
    """

    expected = """
    import anyio
    import threading

    lock = threading.Lock()

    with lock:
        pass
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_asyncio_event(transformer):
    source = """
    import asyncio

    event = asyncio.Event()

    event.set()
    await event.wait()
    assert event.is_set()
    event.clear()
    """

    expected = """
    import asyncio
    import threading

    event = threading.Event()

    event.set()
    event.wait()
    assert event.is_set()
    event.clear()
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_anyio_event(transformer):
    source = """
    import anyio

    event = anyio.Event()

    event.set()
    await event.wait()
    assert event.is_set()
    """

    expected = """
    import anyio
    import threading

    event = threading.Event()

    event.set()
    event.wait()
    assert event.is_set()
    """

    assert transformer(dedent(source)) == dedent(expected)


def test_asyncio_barrier(transformer):
    source = """
    import asyncio

    barrier = asyncio.Barrier(2)

    await barrier.wait()
    await barrier.reset()
    await barrier.abort()

    barrier.parties
    barrier.n_waiting
    barrier.broken
    """

    expected = """
    import asyncio
    import threading

    barrier = threading.Barrier(2)

    barrier.wait()
    barrier.reset()
    barrier.abort()

    barrier.parties
    barrier.n_waiting
    barrier.broken
    """

    assert transformer(dedent(source)) == dedent(expected)
