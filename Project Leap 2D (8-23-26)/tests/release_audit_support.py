from __future__ import annotations

import ast
import hashlib
import importlib.abc
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Iterable, Mapping, Sequence


DEFINITION_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def decorated_start(node: ast.AST) -> int:
    starts = [int(node.lineno)]
    starts.extend(int(item.lineno) for item in getattr(node, "decorator_list", ()))
    return min(starts)


def definition_text(lines: list[str], node: ast.AST) -> str:
    return (
        "".join(lines[decorated_start(node) - 1 : int(node.end_lineno)]).rstrip()
        + "\n"
    )


def top_level_definitions(path: Path) -> dict[str, dict[str, object]]:
    """Return reproducible hashes for top-level functions and classes."""

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    tree = ast.parse("".join(lines), filename=str(path))
    observed: dict[str, dict[str, object]] = {}
    for node in tree.body:
        if not isinstance(node, DEFINITION_TYPES):
            continue
        if node.name in observed:
            raise ValueError(f"Duplicate top-level definition {node.name!r} in {path}")
        text = definition_text(lines, node)
        observed[node.name] = {
            "kind": "class" if isinstance(node, ast.ClassDef) else "function",
            "sha256": sha256_bytes(text.encode("utf-8")),
        }
    return observed


def protected_definition_errors(
    *,
    code_root: Path,
    baseline_manifest: Mapping[str, object],
    registered_additions: Mapping[
        str, Mapping[str, Mapping[str, object]]
    ] | None = None,
    registered_module_sha256: Mapping[str, str] | None = None,
) -> list[str]:
    """Check protected definitions and hash-audit registered additions."""

    expected_rows = baseline_manifest.get("definitions")
    if not isinstance(expected_rows, dict):
        return ["baseline definitions table is missing"]
    additions = {} if registered_additions is None else dict(registered_additions)
    module_hashes = (
        None
        if registered_module_sha256 is None
        else dict(registered_module_sha256)
    )
    errors: list[str] = []
    if module_hashes is not None and set(module_hashes) != set(additions):
        missing = sorted(set(additions) - set(module_hashes))
        extra = sorted(set(module_hashes) - set(additions))
        if missing:
            errors.append(
                f"protected modules with additions have no source hash: {missing}"
            )
        if extra:
            errors.append(
                f"protected module hashes have no registered additions: {extra}"
            )

    by_module: dict[str, dict[str, Mapping[str, object]]] = {}
    for name, raw_row in expected_rows.items():
        if not isinstance(raw_row, dict):
            return [f"baseline definition {name!r} is not an object"]
        module = raw_row.get("module")
        if not isinstance(module, str):
            return [f"baseline definition {name!r} has no module"]
        by_module.setdefault(module, {})[str(name)] = raw_row

    for relative in sorted(set(by_module) | set(additions)):
        expected = by_module.get(relative)
        if expected is None:
            errors.append(
                "registered protected-module addition targets an unprotected "
                "module: "
                f"{relative}"
            )
            continue
        path = code_root / relative
        if not path.is_file():
            errors.append(f"protected module is missing: {relative}")
            continue
        if (
            module_hashes is not None
            and relative in module_hashes
            and sha256_bytes(path.read_bytes()) != module_hashes[relative]
        ):
            errors.append(f"protected module with additions changed: {relative}")
        try:
            observed = top_level_definitions(path)
        except (OSError, SyntaxError, ValueError) as exc:
            errors.append(f"cannot audit protected module {relative}: {exc}")
            continue
        for name, row in sorted(expected.items()):
            item = observed.get(name)
            if item is None:
                errors.append(f"protected definition is missing: {relative}:{name}")
                continue
            if item["kind"] != row.get("kind"):
                errors.append(
                    f"protected definition kind changed: {relative}:{name} "
                    f"{row.get('kind')} -> {item['kind']}"
                )
            if item["sha256"] != row.get("sha256"):
                errors.append(f"protected definition body changed: {relative}:{name}")
        registered = additions.get(relative, {})
        collisions = sorted(set(registered) & set(expected))
        if collisions:
            errors.append(
                f"registered additions collide with protected definitions in "
                f"{relative}: {collisions}"
            )
        observed_additions = {
            name: row for name, row in observed.items() if name not in expected
        }
        if observed_additions != registered:
            missing = sorted(set(registered) - set(observed_additions))
            extra = sorted(set(observed_additions) - set(registered))
            changed = sorted(
                name
                for name in set(observed_additions) & set(registered)
                if observed_additions[name] != registered[name]
            )
            if missing:
                errors.append(
                    f"registered additions missing in {relative}: {missing}"
                )
            if extra:
                errors.append(
                    f"unregistered additions in protected module {relative}: {extra}"
                )
            if changed:
                errors.append(
                    f"registered protected-module additions changed in "
                    f"{relative}: {changed}"
                )
    return errors


def registered_definition_errors(
    *,
    code_root: Path,
    registered_modules: Sequence[str],
    registered: Mapping[str, Mapping[str, Mapping[str, object]]],
    registered_module_sha256: Mapping[str, str] | None = None,
) -> list[str]:
    """Require every registered module and definition to match its hash."""

    errors: list[str] = []
    expected_modules = set(registered_modules)
    modules_with_definitions = set(registered)
    module_hashes = (
        None
        if registered_module_sha256 is None
        else dict(registered_module_sha256)
    )
    if module_hashes is not None:
        missing_hashes = sorted(expected_modules - set(module_hashes))
        extra_hashes = sorted(set(module_hashes) - expected_modules)
        if missing_hashes:
            errors.append(
                f"registered modules have no source hash: {missing_hashes}"
            )
        if extra_hashes:
            errors.append(
                f"unapproved module hashes are registered: {extra_hashes}"
            )
    for relative in sorted(expected_modules | modules_with_definitions):
        if relative not in expected_modules:
            errors.append(f"unapproved module is registered: {relative}")
            continue
        path = code_root / relative
        if not path.is_file():
            errors.append(f"required registered module is missing: {relative}")
            continue
        if (
            module_hashes is not None
            and relative in module_hashes
            and sha256_bytes(path.read_bytes()) != module_hashes[relative]
        ):
            errors.append(f"registered module source changed: {relative}")
        try:
            observed = top_level_definitions(path)
        except (OSError, SyntaxError, ValueError) as exc:
            errors.append(f"cannot audit registered module {relative}: {exc}")
            continue
        expected = registered.get(relative)
        if expected is None:
            errors.append(f"registered module has no definition registry: {relative}")
            continue
        if observed != expected:
            missing = sorted(set(expected) - set(observed))
            extra = sorted(set(observed) - set(expected))
            changed = sorted(
                name
                for name in set(observed) & set(expected)
                if observed[name] != expected[name]
            )
            if missing:
                errors.append(
                    f"registered definitions missing in {relative}: {missing}"
                )
            if extra:
                errors.append(
                    f"unregistered definitions in {relative}: {extra}"
                )
            if changed:
                errors.append(
                    f"registered definitions changed in {relative}: {changed}"
                )
    return errors


def registered_module_surface_errors(
    *,
    code_root: Path,
    registered_modules: Sequence[str],
    registered_module_discovery_globs: Sequence[str],
) -> list[str]:
    """Ensure every source matched by the policy is registered."""

    registered = set(registered_modules)
    discovered: set[str] = set()
    errors: list[str] = []
    for pattern in registered_module_discovery_globs:
        matches = {
            path.relative_to(code_root).as_posix()
            for path in code_root.glob(pattern)
            if path.is_file()
            and path.suffix == ".py"
        }
        if not matches:
            errors.append(f"module discovery glob matched no modules: {pattern}")
        discovered.update(matches)
    missing = sorted(discovered - registered)
    outside = sorted(registered - discovered)
    if missing:
        errors.append(f"modules are not registered: {missing}")
    if outside:
        errors.append(
            f"registered modules are outside the discovery policy: {outside}"
        )
    return errors


def registered_file_hash_errors(
    project_root: Path,
    registered: Mapping[str, str],
) -> list[str]:
    """Hash-audit non-Python resources that alter scientific/runtime behavior."""

    errors: list[str] = []
    for relative, expected in sorted(registered.items()):
        path = project_root / relative
        if not path.is_file():
            errors.append(f"registered resource is missing: {relative}")
            continue
        observed = sha256_bytes(path.read_bytes())
        if observed != expected:
            errors.append(f"registered resource changed: {relative}")
    return errors


def markdown_reference_errors(project_root: Path, allowed_files: Iterable[str]) -> list[str]:
    """Audit the deliberately small human-document surface and local links."""

    allowed = set(allowed_files)
    observed = {
        path.relative_to(project_root).as_posix()
        for path in project_root.rglob("*.md")
        if "__pycache__" not in path.parts
    }
    errors: list[str] = []
    missing = sorted(allowed - observed)
    unexpected = sorted(observed - allowed)
    if missing:
        errors.append(f"required description files are missing: {missing}")
    if unexpected:
        errors.append(f"unexpected description files remain: {unexpected}")

    for relative in sorted(observed & allowed):
        path = project_root / relative
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            errors.append(f"description file is empty: {relative}")
            continue
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().split("#", 1)[0]
            if (
                not target
                or target.startswith(("http://", "https://", "mailto:"))
            ):
                continue
            resolved = (path.parent / target).resolve()
            try:
                resolved.relative_to(project_root.resolve())
            except ValueError:
                errors.append(f"local link escapes package: {relative} -> {raw_target}")
                continue
            if not resolved.exists():
                errors.append(f"broken local link: {relative} -> {raw_target}")
    return errors


def required_document_token_errors(
    project_root: Path,
    required_tokens: Mapping[str, Sequence[str]],
) -> list[str]:
    errors: list[str] = []
    for relative, tokens in sorted(required_tokens.items()):
        path = project_root / relative
        if not path.is_file():
            errors.append(f"cannot inspect missing description file: {relative}")
            continue
        text = path.read_text(encoding="utf-8")
        for token in tokens:
            if token not in text:
                errors.append(f"{relative} does not reference {token!r}")
    return errors


def release_file_manifest(
    project_root: Path,
    *,
    excluded_relative_paths: Iterable[str],
    excluded_directory_names: Iterable[str],
    excluded_suffixes: Iterable[str],
) -> dict[str, dict[str, object]]:
    """Fingerprint every immutable release file without writing to the package."""

    excluded_paths = set(excluded_relative_paths)
    excluded_dirs = set(excluded_directory_names)
    suffixes = tuple(excluded_suffixes)
    observed: dict[str, dict[str, object]] = {}
    for path in sorted(project_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(project_root).as_posix()
        if relative in excluded_paths:
            continue
        if excluded_dirs.intersection(path.relative_to(project_root).parts):
            continue
        if suffixes and path.name.endswith(suffixes):
            continue
        value = path.read_bytes()
        observed[relative] = {
            "bytes": len(value),
            "sha256": sha256_bytes(value),
        }
    return observed


def top_level_optional_import_errors(
    paths: Iterable[Path],
    forbidden_fragments: Sequence[str],
) -> list[str]:
    """Reject eager optional-model imports in the protected eGFP bootstrap path."""

    errors: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
            else:
                continue
            for name in imported:
                if any(fragment in name for fragment in forbidden_fragments):
                    errors.append(f"eager optional import in {path.name}: {name}")
    return errors


class ForbiddenOptionalImport(RuntimeError):
    pass


@dataclass(frozen=True)
class _BlockedImportFinder(importlib.abc.MetaPathFinder):
    prefixes: tuple[str, ...]

    def find_spec(self, fullname, path=None, target=None):
        if any(
            fullname == prefix or fullname.startswith(prefix + ".")
            for prefix in self.prefixes
        ):
            raise ForbiddenOptionalImport(f"optional import was attempted: {fullname}")
        return None


class forbid_optional_imports:
    """Fail a regression run if the protected eGFP route loads optional features."""

    def __init__(self, prefixes: Sequence[str]):
        self.finder = _BlockedImportFinder(tuple(prefixes))

    def __enter__(self):
        import sys

        sys.meta_path.insert(0, self.finder)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        import sys

        if self.finder in sys.meta_path:
            sys.meta_path.remove(self.finder)
        return False


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value
