"""Build ActRouter's ``FunctionInfo`` from Python source, without running it.

A coding agent usually knows a function by file and name, not as a live
object. :func:`function_metadata_from_source` parses the file with
:mod:`ast` -- the code is never imported or executed -- and returns the same
static metadata actdecor sends: name, signature, decorators, docstring.
Never the body.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Optional, Union

#: FunctionInfo field limits (actrouter/schemas.py)
MAX_SIGNATURE = 1024
MAX_DOCSTRING = 8192
MAX_TYPE = 256

FuncNode = Union[ast.FunctionDef, ast.AsyncFunctionDef]


class FunctionNotFound(LookupError):
    pass


def _find(tree: ast.Module, qualname: str) -> FuncNode:
    """``"summarize"`` or ``"Report.summarize"``; a bare name also matches
    a method when it is unambiguous."""
    parts = qualname.split(".")
    matches: list[tuple[str, FuncNode]] = []

    def visit(node: ast.AST, scope: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, [*scope, child.name])
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                matches.append((".".join([*scope, child.name]), child))
                visit(child, [*scope, child.name, "<locals>"])

    visit(tree, [])
    exact = [n for q, n in matches if q == qualname]
    if exact:
        return exact[0]
    by_name = [(q, n) for q, n in matches if q.split(".")[-1] == parts[-1]]
    if len(by_name) == 1:
        return by_name[0][1]
    if not by_name:
        raise FunctionNotFound(f"no function {qualname!r}")
    raise FunctionNotFound(
        f"{qualname!r} is ambiguous: {', '.join(q for q, _ in by_name)}; pass the qualified name"
    )


def _module_name(path: Path, root: Optional[Path]) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve()) if root else Path(path.name)
    except ValueError:
        rel = Path(path.name)
    parts = list(rel.with_suffix("").parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or path.stem


def function_metadata_from_source(
    path: Union[str, Path],
    qualname: str,
    *,
    root: Optional[Union[str, Path]] = None,
    **hints: Any,
) -> dict[str, Any]:
    """The ``function`` payload for ``POST /v1/route`` for `qualname` in
    the file at `path`. `hints` (``task_kind``, ``tags``, budgets, token
    estimates, ``privacy_tier``) are added as given."""
    path = Path(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = _find(tree, qualname)
    returns = ast.unparse(node.returns) if node.returns else None
    signature = f"({ast.unparse(node.args)})" + (f" -> {returns}" if returns else "")
    params = [a.annotation for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]
    docstring = ast.get_docstring(node)
    found_qualname = qualname if "." in qualname else node.name
    meta: dict[str, Any] = {
        "name": node.name,
        "qualname": found_qualname,
        "module": _module_name(path, Path(root) if root else None),
        "signature": signature[:MAX_SIGNATURE],
        "parameter_types": [ast.unparse(a)[:MAX_TYPE] for a in params if a is not None][:64],
        "return_type": returns[:MAX_TYPE] if returns else None,
        "decorators": [ast.unparse(d)[:MAX_TYPE] for d in node.decorator_list][:32],
        "docstring": docstring[:MAX_DOCSTRING] if docstring else None,
    }
    meta.update({k: v for k, v in hints.items() if v is not None})
    return meta
