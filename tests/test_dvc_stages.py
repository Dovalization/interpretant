"""The DVC stages: each lists exactly the modules its pipeline imports (decision agenda, step 9).

A stage that misses a module it imports would not rerun when that module changes; a stage that
lists a module it does not import would rerun for nothing. Package ``__init__.py`` files are left
out: they only re-export, and the modules they import are in the list themselves.
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
PACKAGE = "interpretant"


def module_file(name: str) -> Path | None:
    base = SOURCE.joinpath(*name.split("."))
    if (base / "__init__.py").exists():
        return base / "__init__.py"
    if base.with_suffix(".py").exists():
        return base.with_suffix(".py")
    return None


def imported_modules(file: Path, name: str) -> set[str]:
    """The package's modules that ``file`` imports, absolute and relative imports alike."""
    package = name if file.name == "__init__.py" else name.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1])
                module = f"{base}.{module}" if module else base
            found.add(module)
            found |= {f"{module}.{alias.name}" for alias in node.names}
    return {item for item in found if item.startswith(PACKAGE) and module_file(item)}


def import_closure(root: str) -> set[Path]:
    """Every module file of the package that importing ``root`` loads, ``__init__.py`` aside."""
    seen: set[str] = set()
    pending = [root]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        parts = name.split(".")
        pending += [".".join(parts[:index]) for index in range(1, len(parts))]
        file = module_file(name)
        assert file is not None, name
        pending += imported_modules(file, name)
    files = {module_file(name) for name in seen}
    return {file for file in files if file is not None and file.name != "__init__.py"}


def listed_modules(dependencies: list[str]) -> set[Path]:
    """The Python files a stage's dependencies cover, a listed folder standing for all of its."""
    files: set[Path] = set()
    for dependency in dependencies:
        path = REPOSITORY / dependency
        if path.suffix == ".py":
            files.add(path)
        elif path.is_relative_to(SOURCE) and path.is_dir():
            files |= set(path.rglob("*.py"))
    return {file for file in files if file.name != "__init__.py"}


def stages() -> dict[str, dict[str, object]]:
    document = yaml.safe_load((REPOSITORY / "dvc.yaml").read_text(encoding="utf-8"))
    return dict(document["stages"])


def test_each_stage_runs_its_own_command() -> None:
    for name, stage in stages().items():
        assert stage["cmd"] == f"interpretant {name.replace('_', '-')}"


def test_each_stage_lists_exactly_the_modules_it_imports() -> None:
    for name, stage in stages().items():
        dependencies = stage["deps"]
        assert isinstance(dependencies, list)
        imported = import_closure(f"{PACKAGE}.pipelines.{name}")
        listed = listed_modules(dependencies)
        relative = sorted(str(file.relative_to(REPOSITORY)) for file in imported ^ listed)
        assert listed == imported, f"{name}: listed or imported only once: {relative}"
