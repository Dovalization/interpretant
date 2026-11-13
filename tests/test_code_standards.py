"""The naming rules of docs/code_standards.md that ruff cannot check.

Banned abbreviations, single-letter names, exception variables and test names, read from the
syntax tree of every module in src/ and tests/. Modules written before the standards are exempt
only from the rules they break today (LEGACY); a rebuilt module leaves the list.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CHECKED_FOLDERS = ("src", "tests", "experiments")

BANNED_NAMES = frozenset(
    {
        "cfg", "conn", "req", "resp", "msg", "obj", "arr", "fn", "idx", "tmp", "val", "elem",
        "prev", "nxt", "cur", "df", "vec", "emb", "mat", "res", "ret", "num", "cnt", "lst",
    }
)  # fmt: skip
# Click's context keeps its library name in the command-line interface.
CLI_NAMES = frozenset({"ctx"})
CLI_FILES = ("src/interpretant/__main__.py", "src/interpretant/cli/")
MATHS_LETTERS = frozenset({"i", "j", "k", "n", "m", "x", "y", "_"})
TEST_NAME = re.compile(r"^test_[a-z0-9]+(?:_[a-z0-9]+)+$")

# Modules written before the standards, with the rules each breaks today. Delete an entry when
# the module is rebuilt; never add one.
LEGACY: dict[str, frozenset[str]] = {
    "src/interpretant/__main__.py": frozenset({"exception-name"}),
    "src/interpretant/corpus/openalex.py": frozenset({"exception-name"}),
    "src/interpretant/drift/labels.py": frozenset({"exception-name"}),
    "src/interpretant/drift/pairs.py": frozenset({"single-letter"}),
    "tests/test_drift.py": frozenset({"single-letter"}),
}


def python_files() -> Iterator[Path]:
    for folder in CHECKED_FOLDERS:
        yield from sorted((ROOT / folder).rglob("*.py"))


def bound_names(tree: ast.AST) -> Iterator[tuple[str, int, bool]]:
    """Every name the module binds, with its line and whether a comprehension or lambda binds it."""
    in_scope_of_expression: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.comprehension):
            in_scope_of_expression.update(id(name) for name in ast.walk(node.target))
        elif isinstance(node, ast.Lambda):
            in_scope_of_expression.update(id(argument) for argument in node.args.args)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            yield node.id, node.lineno, id(node) in in_scope_of_expression
        elif isinstance(node, ast.arg):
            yield node.arg, node.lineno, id(node) in in_scope_of_expression
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            yield node.name, node.lineno, False


def violations(source: str, relative: str) -> Iterator[tuple[str, str]]:
    """The rule broken and where, for every naming violation in one module's source."""
    tree = ast.parse(source)
    is_cli = relative.startswith(CLI_FILES)
    for name, line, is_expression_scope in bound_names(tree):
        if name in BANNED_NAMES or (name in CLI_NAMES and not is_cli):
            yield "banned-name", f"{relative}:{line} `{name}`"
        if len(name) == 1 and name not in MATHS_LETTERS and not is_expression_scope:
            yield "single-letter", f"{relative}:{line} `{name}`"
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.name not in (None, "exc"):
            yield "exception-name", f"{relative}:{node.lineno} `{node.name}`"
        if (
            relative.startswith("tests/")
            and isinstance(node, ast.FunctionDef)
            and node.name.startswith("test")
            and not TEST_NAME.match(node.name)
        ):
            yield "test-name", f"{relative}:{node.lineno} `{node.name}`"


@pytest.mark.parametrize(
    "path", list(python_files()), ids=lambda path: path.relative_to(ROOT).as_posix()
)
def test_module_follows_the_naming_rules(path: Path) -> None:
    relative = path.relative_to(ROOT).as_posix()
    exempt = LEGACY.get(relative, frozenset())
    source = path.read_text(encoding="utf-8")
    found = [where for rule, where in violations(source, relative) if rule not in exempt]
    assert not found, "naming rules broken (docs/code_standards.md, Names):\n" + "\n".join(found)


def test_legacy_list_names_only_existing_modules() -> None:
    missing = [module for module in LEGACY if not (ROOT / module).exists()]
    assert not missing, f"remove deleted modules from LEGACY: {missing}"


PLANTED = """
def fetch_rows(cfg):
    for r in cfg.rows:
        pass
    total = [row for row in cfg.rows]
    scale = lambda v: v * 2
    try:
        pass
    except ValueError as error:
        pass
"""


def test_violations_catch_each_kind_of_planted_mistake() -> None:
    found = {(rule, where.split(" ", 1)[1]) for rule, where in violations(PLANTED, "src/m.py")}
    assert ("banned-name", "`cfg`") in found
    assert ("single-letter", "`r`") in found
    assert ("exception-name", "`error`") in found
    assert not any(where in ("`row`", "`v`") for _, where in found)  # comprehension and lambda


def test_violations_check_test_names_only_in_tests() -> None:
    source = "def testBadName():\n    pass\n"
    assert [rule for rule, _ in violations(source, "tests/test_m.py")] == ["test-name"]
    assert list(violations(source, "src/m.py")) == []
