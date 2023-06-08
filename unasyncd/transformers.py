from __future__ import annotations

import itertools
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import libcst as cst
import libcst.matchers as m
from libcst import MetadataWrapper
from libcst.metadata import QualifiedName, ScopeProvider
from libcst.native import parse_module

AnyImport = cst.ImportFrom | cst.Import
ScopedNodeT = TypeVar("ScopedNodeT", bound=cst.Module | cst.FunctionDef | cst.ClassDef)
AnyImportT = TypeVar("AnyImportT", bound=AnyImport)


NAME_REPLACEMENTS = {
    "__aenter__": "__enter__",
    "__aexit__": "__exit__",
    "__anext__": "__next__",
    "__aiter__": "__iter__",
    "athrow": "throw",
    "asend": "send",
    "anext": "next",
    "aiter": "iter",
    "StopAsyncIteration": "StopIteration",
    "contextlib.asynccontextmanager": "contextlib.contextmanager",
    "typing.AsyncIterable": "typing.Iterable",
    "typing.AsyncIterator": "typing.Iterator",
    "typing.AsyncGenerator": "typing.Generator",
    "asyncio.sleep": "time.sleep",
    "anyio.sleep": "time.sleep",
}


UNASYNC_REPLACEMENTS = {
    "async def": "def",
    "async for": "for",
    "async with": "with",
    "await ": "",
    **NAME_REPLACEMENTS,
}


ASYNC_GEN_RE = re.compile(r"AsyncGenerator\[(.+,.+)]")


@dataclass
class TransformerMeta:
    module_imports: dict[str, AnyImport]
    needs_from_import: defaultdict[str, set[str]]
    needs_module_import: set[str]
    exclude: set[tuple[str, ...]]
    removed_names: defaultdict[str, set[str]]


@dataclass
class ImportMeta:
    module_imports: dict[str, cst.Import]
    from_imports: dict[str, cst.ImportFrom]


class StringTransformer:
    """A simple string based transformer"""

    def __init__(self, extra_name_replacements: dict[str, str] | None = None) -> None:
        self.replacements = {**UNASYNC_REPLACEMENTS, **(extra_name_replacements or {})}

    def __call__(self, source: str | bytes) -> str:
        if isinstance(source, bytes):
            source = source.decode()
        output = ASYNC_GEN_RE.sub(r"Generator[\1, None]", source)
        # Substitute 2 argument form of AsyncGenerator with the fully typed equivalent
        # 3 argument Generator form to pass mypy strict

        for string, replacement in self.replacements.items():
            output = output.replace(string, replacement)

        return output


def _create_name_or_attr(qualified_name: str) -> cst.Attribute | cst.Name:
    """Create a name or attribute from a fully qualified name"""
    if "." in qualified_name:
        attributes, qualified_name = qualified_name.rsplit(".", 1)
        return cst.Attribute(
            attr=cst.Name(value=qualified_name), value=_create_name_or_attr(attributes)
        )
    return cst.Name(value=qualified_name)


def _get_docstring_node(
    body: cst.BaseSuite | Sequence[cst.SimpleStatementLine | cst.BaseCompoundStatement],
) -> cst.SimpleString | cst.ConcatenatedString | None:
    """Get the ```SimpleString`` or ``ConcatenatedString`` node of ``body`` representing
    its docstring
    """
    expr: (
        cst.BaseSuite
        | cst.SimpleStatementLine
        | cst.BaseCompoundStatement
        | cst.BaseExpression
        | cst.BaseStatement
        | cst.BaseSmallStatement
    )
    if isinstance(body, Sequence):
        if body:
            expr = body[0]
        else:
            return None
    else:
        expr = body

    while isinstance(expr, cst.BaseSuite | cst.SimpleStatementLine):
        if len(expr.body) == 0:
            return None

        expr = expr.body[0]
    if not isinstance(expr, cst.Expr):
        return None

    val = expr.value
    if isinstance(val, cst.SimpleString | cst.ConcatenatedString):
        return val

    return None


class TreeTransformer:
    """libcst based transformer"""

    def __init__(
        self,
        exclude: Iterable[str] | None = None,
        transform_docstrings: bool = True,
        extra_name_replacements: dict[str, str] | None = None,
        remove_unused_imports: bool = False,
    ) -> None:
        self.exclude = exclude
        self.transform_docstrings = transform_docstrings
        self.name_replacements = {
            **NAME_REPLACEMENTS,
            **(extra_name_replacements or {}),
        }
        self.remove_unused_imports = remove_unused_imports

    def __call__(self, source: str) -> str:
        if not source:
            return ""

        wrapper = MetadataWrapper(parse_module(source), unsafe_skip_copy=True)
        result = wrapper.visit(
            _AsyncTransformer(
                exclude=self.exclude,
                name_replacements=self.name_replacements,
                transform_docstrings=self.transform_docstrings,
                remove_unused_imports=self.remove_unused_imports,
            )
        )
        output = result.code

        # newlines at the end can get lost so ensure we have one
        if source[-1] == result.default_newline and not result.has_trailing_newline:
            output += result.default_newline
        return output


class _ReplaceAttributesMixin:
    """
    ``CSTTransformer`` mixin that replaces attributes by their fully qualified name
    """

    _replace_attributes: dict[str, dict[str, str]]

    def leave_Attribute(
        self, original_node: cst.Attribute, updated_node: cst.Attribute
    ) -> cst.BaseExpression:
        maybe_name = updated_node.value
        if not isinstance(maybe_name, cst.Name):
            return updated_node
        name = maybe_name.value

        if not (attribute_replacements := self._replace_attributes.get(name)):
            return updated_node

        attr_name = updated_node.attr.value

        if attr_replacement := attribute_replacements.get(attr_name):
            updated_node = updated_node.with_changes(
                attr=_create_name_or_attr(attr_replacement)
            )

        return updated_node


class _ReplaceNamesMixin:
    """``CSTTransformer`` mixin that replaces names by their fully qualified name"""

    _in_import: bool = False
    _name_replacements: dict[str, str]
    meta: TransformerMeta

    def _create_replacement_name_or_attribute_for(
        self, node: cst.Name | cst.Attribute, replace_with: str
    ) -> cst.Name | cst.Attribute:
        """Create a replacement name or attribute for the given note"""
        if isinstance(node, cst.Name):
            if "." in replace_with:
                module_name, attr = replace_with.rsplit(".", 1)
                self.meta.needs_from_import[module_name].add(attr)
                return cst.Name(attr)
            return cst.Name(replace_with)

        self.meta.needs_module_import.add(replace_with.rsplit(".", 1)[0])
        return _create_name_or_attr(replace_with)

    def leave_Name(
        self, original_node: cst.Name, updated_node: cst.Name
    ) -> cst.BaseExpression:
        if self._in_import:
            return updated_node

        old_name: str = updated_node.value
        new_name: str | None = None

        try:
            scope = self.get_metadata(  # type: ignore[attr-defined]
                ScopeProvider,
                original_node,
            )
            qualified_names: set[QualifiedName] = scope.get_qualified_names_for(
                original_node
            )

            for _name in qualified_names:
                new_name = self._name_replacements.get(_name.name)
                if new_name:
                    old_name = _name.name
                    break
            if new_name and len(qualified_names) > 1:
                raise RuntimeError("Could not determine fully qualified name")

        except KeyError:
            pass

        if not new_name:
            new_name = self._name_replacements.get(old_name)

        if new_name:
            if (
                "." in old_name
            ):  # if not a fully qualified name we don't care about imports
                module_name, attr = old_name.rsplit(".", 1)
                self.meta.removed_names[module_name].add(attr)

            return self._create_replacement_name_or_attribute_for(
                updated_node, new_name
            )

        return updated_node


class ContextManagerTransformer(cst.CSTTransformer):
    def __init__(self, *, asname: str, meta: TransformerMeta) -> None:
        super().__init__()
        self.name = asname
        self.meta = meta


class _AsyncExitStackBodyTransformer(
    _ReplaceAttributesMixin, ContextManagerTransformer
):
    """
    ``CSTTransformer`` transforming a ``contextlib.AsyncExitStack into a
    ``contextlib.ExitStack``
    """

    def __init__(self, *, asname: str, meta: TransformerMeta) -> None:
        super().__init__(asname=asname, meta=meta)
        self._replace_attributes = {
            self.name: {
                "enter_async_context": "enter_context",
                "push_async_exit": "push",
                "push_async_callback": "callback",
                "aclose": "close",
            }
        }


class _AnyioTaskGroupBodyTransformer(
    _ReplaceAttributesMixin, _ReplaceNamesMixin, ContextManagerTransformer
):
    """
    ``CSTTransformer`` transforming ``anyio.create_task_group()`` into usages of
    ``concurrent.futures.ThreadPoolExecutor``
    """

    def __init__(self, *, asname: str, meta: TransformerMeta) -> None:
        self._name_replacements = {}
        if asname in {"task_group", "tg"}:
            self._name_replacements["task_group"] = "executor"
            asname = "executor"
        self._replace_attributes = {asname: {"start_soon": "submit", "start": "submit"}}

        super().__init__(asname=asname, meta=meta)


class _TaskGroupBodyTransformer(
    _ReplaceAttributesMixin, _ReplaceNamesMixin, ContextManagerTransformer
):
    """
    ``CSTTransformer`` transforming ``asyncio.TaskGroup()`` into usages of
    ``concurrent.futures.ThreadPoolExecutor``
    """

    def __init__(self, *, asname: str, meta: TransformerMeta) -> None:
        super().__init__(asname=asname, meta=meta)
        self.call_matcher = m.Call(
            func=m.Attribute(
                value=m.Name(value=self.name), attr=m.Name(value="create_task")
            )
        )

        self._name_replacements = {}

        if self.name in {"task_group", "tg"}:
            self._name_replacements["task_group"] = "executor"
            self.name = "executor"

        self._replace_attributes = {self.name: {"create_task": "submit"}}

    def leave_Call(
        self, original_node: cst.Call, updated_node: cst.Call
    ) -> cst.BaseExpression:
        if not m.matches(original_node, self.call_matcher):
            return updated_node

        first_arg = updated_node.args[0]
        if not m.matches(first_arg, m.Arg(value=m.Call(m.DoNotCare()))):
            return updated_node

        new_arg = first_arg.with_changes(
            value=cst.ensure_type(first_arg.value, cst.Call).func
        )
        return updated_node.with_changes(args=[new_arg, *updated_node.args[1:]])


class _AsyncTransformer(_ReplaceNamesMixin, cst.CSTTransformer):
    """Main entry point for unasyncd"""

    METADATA_DEPENDENCIES = (ScopeProvider,)

    def __init__(
        self,
        *,
        exclude: Iterable[str] | None = None,
        name_replacements: dict[str, str],
        transform_docstrings: bool,
        remove_unused_imports: bool,
    ):
        """

        :param exclude: Iterable of fully qualified names of classes, methods and
            functions to exclude
        :param name_replacements: Mapping of fully qualified names to their replacements
        :param transform_docstrings: If ``True``, transform docstrings
        :param remove_unused_imports:  If ``True`` remove module level imports that have
            become unused due to the transformation
        """
        super().__init__()
        self._current_scop_name: tuple[str, ...] = tuple()
        self._name_replacements = name_replacements
        self._should_transform_docstrings = transform_docstrings
        self.meta = TransformerMeta(
            module_imports={},
            needs_from_import=defaultdict(set),
            exclude={tuple(p.split(".")) for p in exclude or []},
            removed_names=defaultdict(set),
            needs_module_import=set(),
        )
        self._in_import = False
        self.attribute_replacements = {
            name: replacement
            for name, replacement in name_replacements.items()
            if "." in name
        }
        self.expressions_to_remove: set[cst.Expr] = set()
        self.scoped_node_imports: defaultdict[cst.CSTNode, ImportMeta] = defaultdict(
            lambda: ImportMeta(module_imports={}, from_imports={})
        )
        self._scope_nodes: list[cst.CSTNode] = []
        self.remove_unused_imports = remove_unused_imports
        self._string_transformer = StringTransformer()

    @property
    def current_scope_node(self) -> cst.CSTNode | None:
        """The parent node of the current scope (excludes comprehension scope)"""
        if self._scope_nodes:
            return self._scope_nodes[-1]
        return None

    @property
    def _should_transform_current_node(self) -> bool:
        return self._current_scop_name not in self.meta.exclude

    def get_qualified_name(self, node: cst.CSTNode) -> str | None:
        """Get the fully qualified name of ``node``, relative to the current module."""
        try:
            if not (scope := self.get_metadata(ScopeProvider, node)):
                return None

            qualified_names: set[QualifiedName] = set(
                scope.get_qualified_names_for(node)
            )

            if len(qualified_names) == 1:
                return qualified_names.pop().name
            elif qualified_names:  # multiple names found
                raise ValueError(
                    f"Could not determine fully qualified name for {node!r}"
                )

        except KeyError:
            pass

        return None

    def visit_Module(self, node: cst.Module) -> bool:
        self._scope_nodes.append(node)
        return True

    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:
        self._current_scop_name = (*self._current_scop_name, node.name.value)
        self._scope_nodes.append(node)
        return self._should_transform_current_node

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> (
        cst.BaseStatement | cst.FlattenSentinel[cst.BaseStatement] | cst.RemovalSentinel
    ):
        self._current_scop_name = self._current_scop_name[:-1]
        self._scope_nodes = self._scope_nodes[:-1]
        if self._should_transform_docstrings and self._should_transform_current_node:
            updated_node = self._transform_docstring(updated_node)
        return updated_node

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        self._current_scop_name = (*self._current_scop_name, node.name.value)
        self._scope_nodes.append(node)
        return self._should_transform_current_node

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> (
        cst.BaseStatement | cst.FlattenSentinel[cst.BaseStatement] | cst.RemovalSentinel
    ):
        if self._should_transform_current_node:
            if updated_node.asynchronous:
                updated_node = updated_node.with_changes(asynchronous=None)

            if self._should_transform_docstrings:
                updated_node = self._transform_docstring(updated_node)

        self._current_scop_name = self._current_scop_name[:-1]
        self._scope_nodes = self._scope_nodes[:-1]
        return updated_node

    def _transform_docstring(self, updated_node: ScopedNodeT) -> ScopedNodeT:
        if not (docstring_node := _get_docstring_node(updated_node.body)):
            return updated_node

        if not (original_docstring := docstring_node.evaluated_value):
            return updated_node

        updated_docstring = self._string_transformer(original_docstring)
        quote = (
            docstring_node.quote
            if isinstance(docstring_node, cst.SimpleString)
            else docstring_node.left.quote
        )

        if updated_docstring != original_docstring:
            updated_node = updated_node.with_deep_changes(  # type: ignore[assignment]
                docstring_node,
                value=quote + updated_docstring + quote,
            )

        return updated_node

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
        self._in_import = True
        return True

    def visit_Import(self, node: cst.Import) -> bool | None:
        self._in_import = True
        return None

    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> (
        cst.BaseSmallStatement
        | cst.FlattenSentinel[cst.BaseSmallStatement]
        | cst.RemovalSentinel
    ):
        self._in_import = False
        self.register_import(updated_node)
        return updated_node

    def leave_Import(
        self, original_node: cst.Import, updated_node: cst.Import
    ) -> (
        cst.BaseSmallStatement
        | cst.FlattenSentinel[cst.BaseSmallStatement]
        | cst.RemovalSentinel
    ):
        self._in_import = False
        self.register_import(updated_node)
        return updated_node

    def register_import(self, node: cst.Import | cst.ImportFrom) -> None:
        if not self.current_scope_node:
            return

        node_imports = self.scoped_node_imports[self.current_scope_node]
        if isinstance(node, cst.Import):
            for name in node.names:
                node_imports.module_imports[name.evaluated_name] = node
            return None

        module_name = (
            cst.helpers.get_full_name_for_node_or_raise(node.module)
            if node.module
            else "." * len(node.relative)
        )
        node_imports.from_imports[module_name] = node

    def leave_Attribute(
        self, original_node: cst.Attribute, updated_node: cst.Attribute
    ) -> cst.BaseExpression:
        if not (qualified_name := self.get_qualified_name(original_node)):
            return updated_node

        if not (replacement := self.attribute_replacements.get(qualified_name)):
            return updated_node

        module_name, attr = qualified_name.rsplit(".", 1)
        self.meta.removed_names[module_name].add(attr)
        self.meta.needs_module_import.add(replacement.rsplit(".", 1)[0])

        return _create_name_or_attr(replacement)

    def _transform_context_manager(
        self,
        *,
        updated_node: cst.With,
        with_item: cst.WithItem,
        item_call_func: cst.Name | cst.Attribute,
        visitor_class: type[ContextManagerTransformer],
        replace_call: str,
    ) -> cst.With:
        """Transform a context manager's with-item"""
        if not with_item.asname:
            raise ValueError(f"Missing as-name for context manager item {with_item}")

        new_call_func = self._create_replacement_name_or_attribute_for(
            item_call_func, replace_call
        )

        updated_node = updated_node.with_deep_changes(
            with_item, item=cst.Call(func=new_call_func)
        )
        asname = cst.ensure_type(with_item.asname.name, cst.Name)  # can be a tuple,
        # but in none of the supported cases, so we ensure its type here
        visitor = visitor_class(asname=asname.value, meta=self.meta)
        visitor.get_metadata = self.get_metadata  # type: ignore[method-assign]
        return cst.ensure_type(updated_node.visit(visitor), cst.With)

    def _transform_context_managers(
        self,
        original_node: cst.With,
        updated_node: cst.With,
        with_item: cst.WithItem,
    ) -> cst.With:
        """Transform asynchronous context managers that need special handling, such as
        ``asyncio.TaskGroup``
        """
        if not m.matches(
            with_item, m.WithItem(m.Call(func=m.OneOf(m.Name(), m.Attribute())))
        ):
            return updated_node

        item_call_func: cst.Attribute | cst.Name = with_item.item.func  # type: ignore[attr-defined]  # noqa: E501

        if not (scope := self.get_metadata(ScopeProvider, original_node)):
            raise RuntimeError("Scope not found")

        qualified_names = set(scope.get_qualified_names_for(item_call_func))

        for name in qualified_names:
            if name.name == "contextlib.AsyncExitStack":
                self.meta.removed_names["contextlib"].add("AsyncExitStack")
                return self._transform_context_manager(
                    updated_node=updated_node,
                    with_item=with_item,
                    replace_call="contextlib.ExitStack",
                    visitor_class=_AsyncExitStackBodyTransformer,
                    item_call_func=item_call_func,
                )

            if name.name == "asyncio.TaskGroup":
                self.meta.removed_names["asyncio"].add("TaskGroup")
                return self._transform_context_manager(
                    updated_node=updated_node,
                    with_item=with_item,
                    replace_call="concurrent.futures.ThreadPoolExecutor",
                    visitor_class=_TaskGroupBodyTransformer,
                    item_call_func=item_call_func,
                )

            if name.name == "anyio.create_task_group":
                self.meta.removed_names["anyio"].add("create_task_group")
                return self._transform_context_manager(
                    updated_node=updated_node,
                    with_item=with_item,
                    replace_call="concurrent.futures.ThreadPoolExecutor",
                    visitor_class=_AnyioTaskGroupBodyTransformer,
                    item_call_func=item_call_func,
                )

        return updated_node

    def leave_With(
        self, original_node: cst.With, updated_node: cst.With
    ) -> (
        cst.BaseStatement | cst.FlattenSentinel[cst.BaseStatement] | cst.RemovalSentinel
    ):
        if not updated_node.asynchronous:
            return updated_node

        for with_item in updated_node.items:
            updated_node = self._transform_context_managers(
                original_node, updated_node, with_item
            )
        return updated_node.with_changes(asynchronous=None)

    def visit_Expr(self, node: cst.Expr) -> bool | None:
        """Register expressions to be removed entirely. Currently removes
        ``await asyncio.sleep(0)`` and ``await anyio.sleep(0)``
        """
        if m.matches(node, m.Expr(m.Await(expression=m.Call(m.DoNotCare())))):
            call = cst.ensure_type(node.value, cst.Await).expression
            qualified_name = self.get_qualified_name(call)
            if qualified_name not in {"asyncio.sleep", "anyio.sleep"}:
                return True

            if m.matches(
                call, m.Call(args=[m.AtLeastN(m.Arg(m.Integer(value="0")), n=1)])
            ):
                self.expressions_to_remove.add(node)
                removed_mod, removed_name = qualified_name.split(".")
                self.meta.removed_names[removed_mod].add(removed_name)
                return False
        return True

    def leave_Expr(
        self, original_node: cst.Expr, updated_node: cst.Expr
    ) -> cst.BaseSmallStatement | cst.RemovalSentinel:
        """Remove expressions registered to be removed"""
        if original_node in self.expressions_to_remove:
            return cst.RemoveFromParent()
        return updated_node

    def leave_Await(
        self, original_node: cst.Await, updated_node: cst.Await
    ) -> cst.BaseExpression | Any:
        """Unwrap await: ``await foo`` > ``foo``"""
        return updated_node.expression.with_changes(
            lpar=updated_node.lpar,
            rpar=updated_node.rpar,
        )

    def leave_For(
        self, original_node: cst.For, updated_node: cst.For
    ) -> (
        cst.BaseStatement | cst.FlattenSentinel[cst.BaseStatement] | cst.RemovalSentinel
    ):
        """Replace ``async for i in thing`` > ``for i in thing"""
        if updated_node.asynchronous:
            updated_node = updated_node.with_changes(asynchronous=None)
        return updated_node

    def leave_Subscript(
        self, original_node: cst.Subscript, updated_node: cst.Subscript
    ) -> cst.Subscript:
        """Fix ``typing.Generator`` generic form by adding a third ``None`` as a last
        subscript element: ``typing.AsyncGenerator[None, None]`` >
        ``typing.Generator[None, None, None]``.
        """
        if self.get_qualified_name(original_node) != "typing.AsyncGenerator":
            return updated_node

        none_element = cst.SubscriptElement(slice=cst.Index(cst.Name("None")))
        updated_node = updated_node.with_changes(
            slice=[*updated_node.slice, none_element]
        )
        return updated_node

    def _get_imports_from_module(
        self, wrapper: MetadataWrapper
    ) -> tuple[
        dict[str, cst.Import], dict[str, cst.ImportFrom], dict[AnyImport, set[str]]
    ]:
        """Gather information about module level imports.

        Return a tuple of:
            - ``Import`` statements
            - ``ImportFrom`` statements
            - Imported names that have become unused due to the applied transformations
        """
        from_imports: dict[str, cst.ImportFrom] = {}
        module_imports: dict[str, cst.Import] = {}
        imports_to_remove: defaultdict[AnyImport, set[str]] = defaultdict(set)
        scopes = set(
            s
            for s in wrapper.resolve(ScopeProvider).values()
            if isinstance(s, cst.metadata.GlobalScope)
        )
        scope = scopes.pop()

        for assignment in scope.assignments:
            if not isinstance(assignment, cst.metadata.Assignment):
                continue

            node = assignment.node

            if isinstance(node, cst.Import):
                module_imports[assignment.name] = node

                if assignment.references:
                    continue

                removed_names = self.meta.removed_names.get(assignment.name)
                if removed_names:
                    imports_to_remove[node].add(assignment.name)

            elif isinstance(node, cst.ImportFrom):
                module_name = (
                    cst.helpers.get_full_name_for_node_or_raise(node.module)
                    if node.module
                    else "." * len(node.relative)
                )
                from_imports[module_name] = node

                if assignment.references:
                    continue

                if not (removed_names := self.meta.removed_names.get(module_name)):
                    continue

                if isinstance(node.names, cst.ImportStar):
                    continue

                imported_names = {alias.evaluated_name for alias in node.names}
                if names_to_remove := imported_names.intersection(removed_names):
                    imports_to_remove[node].update(names_to_remove)

        return module_imports, from_imports, imports_to_remove

    def _find_first_non_import_line(self, updated_node: cst.Module) -> int:
        """Get the index of the first line of the module that is not an import,
        skipping module level docstrings and comments.
        """
        import_matcher = m.SimpleStatementLine(
            body=[m.AtLeastN(m.OneOf(m.Import(), m.ImportFrom()), n=1)]
        )
        string_matcher = m.SimpleStatementLine([m.Expr(m.SimpleString())])
        for i, node in enumerate(updated_node.body):
            if m.matches(node, string_matcher):
                continue
            if not m.matches(node, import_matcher):
                return i
        return 0

    def _create_from_imports(
        self, existing_from_imports: dict[str, cst.ImportFrom]
    ) -> tuple[dict[cst.ImportFrom, set[str]], list[cst.ImportFrom]]:
        """Return a tuple of ``FromImport``s to update and ``FromImport``s to add"""
        from_imports_to_update: defaultdict[cst.ImportFrom, set[str]] = defaultdict(set)
        from_imports_to_add: defaultdict[str, set[str]] = defaultdict(set)

        for module_name, names in self.meta.needs_from_import.items():
            if _import := existing_from_imports.get(module_name):
                if isinstance(_import.names, cst.ImportStar):
                    continue
                existing_names = {
                    alias.evaluated_alias or alias.evaluated_name
                    for alias in _import.names
                }
                from_imports_to_update[_import].update(names - existing_names)
            else:
                from_imports_to_add[module_name].update(names)

        new_from_imports = [
            cst.ImportFrom(
                module=_create_name_or_attr(module_name),
                names=[cst.ImportAlias(cst.Name(name)) for name in names],
            )
            for module_name, names in from_imports_to_add.items()
        ]
        return from_imports_to_update, new_from_imports

    def _create_module_imports(
        self, existing_module_import: dict[str, cst.Import]
    ) -> list[cst.Import]:
        """Return a list of ``Import``s to add"""
        return [
            cst.Import(names=[cst.ImportAlias(name=_create_name_or_attr(module_name))])
            for module_name in self.meta.needs_module_import
            if not (_import := existing_module_import.get(module_name))
            and not isinstance(_import, cst.ImportFrom)
        ]

    def _fix_module_imports(
        self,
        *,
        node: cst.Module | MetadataWrapper,
        from_imports: dict[str, cst.ImportFrom],
        module_imports: dict[str, cst.Import],
        imports_to_remove: dict[AnyImport, set[str]],
    ) -> cst.Module:
        """Modify a module's imports.

        - Update existing imports with names that have been requested by a
          transformation
        - Add new imports that have not been covered in the previous step
        - Remove names from imports specified in ``imports_to_remove``. If an import has
          no names left, it is removed entirely
        """
        from_imports_to_update, new_from_imports = self._create_from_imports(
            from_imports
        )

        updated_node = node.visit(
            _ImportTransformer(
                unused_imports=imports_to_remove,
                imports_to_update=from_imports_to_update,
            )
        )

        new_module_imports = self._create_module_imports(module_imports)
        import_offset = self._find_first_non_import_line(updated_node)
        new_module_body = [
            *updated_node.body[:import_offset],
            *[
                cst.SimpleStatementLine(body=[new_import])
                for new_import in itertools.chain(new_module_imports, new_from_imports)
            ],
            *updated_node.body[import_offset:],
        ]
        updated_node = updated_node.with_changes(body=new_module_body)

        if self._should_transform_docstrings and self._should_transform_current_node:
            updated_node = self._transform_docstring(updated_node)

        return updated_node

    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        if self.remove_unused_imports and self.meta.removed_names:
            wrapper = MetadataWrapper(updated_node)
            (
                module_imports,
                from_imports,
                imports_to_remove,
            ) = self._get_imports_from_module(wrapper)
            return self._fix_module_imports(
                node=wrapper,
                module_imports=module_imports,
                from_imports=from_imports,
                imports_to_remove=imports_to_remove,
            )

        import_meta = self.scoped_node_imports[original_node]
        return self._fix_module_imports(
            node=updated_node,
            module_imports=import_meta.module_imports,
            from_imports=import_meta.from_imports,
            imports_to_remove={},
        )


class _ImportTransformer(cst.CSTTransformer):
    # taken from
    # https://cst.readthedocs.io/en/latest/scope_tutorial.html#Automatically-Remove-Unused-Import

    def __init__(
        self,
        *,
        unused_imports: dict[AnyImport, set[str]],
        imports_to_update: dict[cst.ImportFrom, set[str]],
    ) -> None:
        super().__init__()
        self.unused_imports = unused_imports
        self.imports_to_update = imports_to_update

    def leave_import_alike(
        self, original_node: AnyImportT, updated_node: AnyImportT
    ) -> AnyImportT | cst.RemovalSentinel:
        if original_node not in self.unused_imports or isinstance(
            updated_node.names, cst.ImportStar
        ):
            return updated_node

        names_to_keep = []
        for name in updated_node.names:
            name_value = name.name.value

            if name_value not in self.unused_imports[original_node]:
                names_to_keep.append(name.with_changes(comma=cst.MaybeSentinel.DEFAULT))

        if len(names_to_keep) == 0:
            return cst.RemoveFromParent()
        else:
            return updated_node.with_changes(names=names_to_keep)  # type: ignore[return-value]  # noqa: E501

    def leave_Import(
        self, original_node: cst.Import, updated_node: cst.Import
    ) -> cst.Import | cst.RemovalSentinel:
        return self.leave_import_alike(original_node, updated_node)

    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> cst.ImportFrom | cst.RemovalSentinel:
        aliases_to_add = self.imports_to_update.get(original_node)
        if aliases_to_add:
            new_aliases = [cst.ImportAlias(cst.Name(name)) for name in aliases_to_add]

            current_names = (
                updated_node.names
                if not isinstance(updated_node.names, cst.ImportStar)
                else [updated_node.names]  # type: ignore[list-item]
            )

            updated_node = updated_node.with_changes(
                names=[*current_names, *new_aliases]
            )

        return self.leave_import_alike(original_node, updated_node)

    # since we're only interested in module level imports, we don't need to visit
    # class and function definitions

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        return False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        return False
