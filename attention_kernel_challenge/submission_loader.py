from __future__ import annotations

import ast
import builtins
import hashlib
import importlib
import importlib.util
import inspect
import io
import os
import sys
import sysconfig
import tarfile
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterator, List, Optional, Sequence, Set

import torch

from .spec import MAX_VARIANT_COUNT, CaseSpec, FamilyName, VariantSpec

ALLOWED_THIRD_PARTY = {"torch", "triton", "numpy"}
COMPILER_TRANSITIVE_ROOTS = {
    "typing_extensions",
    "sympy",
    "mpmath",
    "filelock",
    "jinja2",
    "packaging",
    "networkx",
    "colorama",
    "functorch",
    "torchgen",
}
MAX_SUBMISSION_ARCHIVE_BYTES = 10 * 1024 * 1024
MAX_SUBMISSION_FILE_COUNT = 128
MAX_SUBMISSION_UNPACKED_BYTES = 20 * 1024 * 1024
ALLOWED_SUBMISSION_SUFFIXES = {".py", ".json", ".toml", ".txt", ".md"}


@dataclass
class LoadedSubmission:
    entrypoint: Callable
    setup: Optional[Callable]
    variants: tuple[VariantSpec, ...]
    module_name: str
    root_dir: Path

    def run_setup(self, suite: List[CaseSpec], device: str) -> None:
        if self.setup is None:
            return

        signature = inspect.signature(self.setup)
        if len(signature.parameters) == 0:
            self.setup()
            return
        if len(signature.parameters) == 2:
            self.setup(suite, device)
            return
        if len(signature.parameters) == 3:
            self.setup(suite, device, self.variants)
            return
        raise TypeError(
            "Optional submission setup() must accept either no arguments or "
            "(suite_specs, device) or (suite_specs, device, variants)."
        )


def load_submission(submission_dir: str | os.PathLike[str]) -> LoadedSubmission:
    root_dir = Path(submission_dir).resolve()
    submission_file = root_dir / "submission.py"
    if not submission_file.exists():
        raise FileNotFoundError(f"Expected submission.py in {root_dir}")

    _validate_python_files(root_dir)
    unique_module_name = f"contest_submission_{hashlib.sha256(str(root_dir).encode()).hexdigest()[:16]}"
    local_modules = _discover_local_top_level_modules(root_dir)

    with _prepend_sys_path(root_dir), _submission_import_guard(local_modules):
        spec = importlib.util.spec_from_file_location(unique_module_name, submission_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create import spec for {submission_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(unique_module_name, None)
            raise

    entrypoint = getattr(module, "block_sparse_attn_fwd", None)
    if entrypoint is None or not callable(entrypoint):
        raise AttributeError("submission.py must define callable block_sparse_attn_fwd")
    setup = getattr(module, "setup", None)
    if setup is not None and not callable(setup):
        raise TypeError("If provided, setup must be callable")
    variants = _load_variant_manifest(module)
    entrypoint = _wrap_submission_callable(entrypoint, local_modules)
    if setup is not None:
        setup = _wrap_submission_callable(setup, local_modules)

    return LoadedSubmission(
        entrypoint=entrypoint,
        setup=setup,
        variants=variants,
        module_name=unique_module_name,
        root_dir=root_dir,
    )


def pack_submission_dir(
    submission_dir: str | os.PathLike[str],
    max_archive_bytes: int = MAX_SUBMISSION_ARCHIVE_BYTES,
) -> bytes:
    root_dir = Path(submission_dir).resolve()
    submission_file = root_dir / "submission.py"
    if not submission_file.exists():
        raise FileNotFoundError(f"Expected submission.py in {root_dir}")
    _validate_submission_files(root_dir)

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(_submission_files(root_dir)):
            arcname = path.relative_to(root_dir)
            archive.add(path, arcname=str(arcname), recursive=False)
    payload = buffer.getvalue()
    if len(payload) > max_archive_bytes:
        raise ValueError(
            f"Compressed submission archive is too large: {len(payload)} bytes > {max_archive_bytes} bytes"
        )
    return payload


def unpack_submission_archive(archive_bytes: bytes) -> Path:
    root_dir = Path(tempfile.mkdtemp(prefix="attention-kernel-challenge-submission-"))
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk():
                raise ValueError("Submission archive may not contain symlinks.")
            if member.isdir():
                continue
            member_path = root_dir / member.name
            _ensure_safe_extract_path(root_dir, member_path)
        archive.extractall(root_dir)
    _validate_submission_files(root_dir)
    return root_dir


def _submission_files(root_dir: Path) -> Iterator[Path]:
    for path in root_dir.rglob("*"):
        if path.is_dir():
            continue
        if path.name == ".DS_Store":
            continue
        if "__pycache__" in path.parts:
            continue
        yield path


def _validate_submission_files(root_dir: Path) -> None:
    total_bytes = 0
    file_count = 0
    for path in _submission_files(root_dir):
        file_count += 1
        if file_count > MAX_SUBMISSION_FILE_COUNT:
            raise ValueError(
                f"Submission contains too many files: {file_count} > {MAX_SUBMISSION_FILE_COUNT}"
            )
        if path.suffix not in ALLOWED_SUBMISSION_SUFFIXES:
            raise ValueError(
                f"Disallowed submission file type for {path.name!r}. "
                f"Allowed suffixes are: {sorted(ALLOWED_SUBMISSION_SUFFIXES)}"
            )
        total_bytes += path.stat().st_size
        if total_bytes > MAX_SUBMISSION_UNPACKED_BYTES:
            raise ValueError(
                f"Submission contents are too large: {total_bytes} bytes > {MAX_SUBMISSION_UNPACKED_BYTES} bytes"
            )


def _validate_python_files(root_dir: Path) -> None:
    _validate_submission_files(root_dir)
    local_modules = _discover_local_top_level_modules(root_dir)
    for path in _submission_files(root_dir):
        if path.suffix != ".py":
            continue
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    _validate_import_name(alias.name, local_modules, path)
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    continue
                if node.module is None:
                    continue
                _validate_import_name(node.module, local_modules, path)


def _validate_import_name(module_name: str, local_modules: Set[str], path: Path) -> None:
    top_level = module_name.split(".", 1)[0]
    if top_level in local_modules:
        return
    if top_level in ALLOWED_THIRD_PARTY:
        return
    if _is_stdlib_module(top_level):
        return
    raise ImportError(
        f"Disallowed third-party import {module_name!r} in {path}. "
        f"Allowed third-party modules are: {sorted(ALLOWED_THIRD_PARTY)}"
    )


def _discover_local_top_level_modules(root_dir: Path) -> Set[str]:
    modules: Set[str] = {"submission"}
    for path in root_dir.iterdir():
        if path.name.startswith("."):
            continue
        if path.is_file() and path.suffix == ".py":
            modules.add(path.stem)
        elif path.is_dir() and (path / "__init__.py").exists():
            modules.add(path.name)
    return modules


def _is_stdlib_module(module_name: str) -> bool:
    if module_name in sys.builtin_module_names:
        return True
    stdlib_names = getattr(sys, "stdlib_module_names", None)
    if stdlib_names is not None and module_name in stdlib_names:
        return True

    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return False
    if spec.origin in {"built-in", "frozen"}:
        return True

    stdlib_path = Path(sysconfig.get_paths()["stdlib"]).resolve()
    origin_path = Path(spec.origin).resolve()
    try:
        origin_path.relative_to(stdlib_path)
    except ValueError:
        return False
    return "site-packages" not in origin_path.parts


def _ensure_safe_extract_path(root_dir: Path, path: Path) -> None:
    resolved_root = root_dir.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Unsafe archive path: {path}") from exc


@contextmanager
def _prepend_sys_path(path: Path) -> Iterator[None]:
    original = list(sys.path)
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path[:] = original


@contextmanager
def _submission_import_guard(local_modules: Set[str]) -> Iterator[None]:
    original_import = builtins.__import__
    original_import_module = importlib.import_module
    trusted_depth = 0
    trusted_modules: dict[str, tuple[int, str | None]] = {}

    def _normalize_origin(origin: object) -> str | None:
        if origin is None:
            return None
        candidate = os.fspath(origin)
        if candidate.startswith("<"):
            return candidate
        try:
            return str(Path(candidate).resolve())
        except OSError:
            return candidate

    def _module_origin(module: ModuleType) -> str | None:
        module_file = getattr(module, "__file__", None)
        if module_file is not None:
            return _normalize_origin(module_file)
        module_spec = getattr(module, "__spec__", None)
        if module_spec is not None:
            return _normalize_origin(getattr(module_spec, "origin", None))
        return None

    def _resolve_importer_context(globals_dict: dict | None) -> tuple[str | None, dict | None, str | None]:
        importer_name = globals_dict.get("__name__") if globals_dict is not None else None
        importer_globals = globals_dict
        importer_filename = globals_dict.get("__file__") if globals_dict is not None else None
        frame = inspect.currentframe()
        caller = frame.f_back.f_back if frame is not None and frame.f_back is not None else None
        if caller is not None:
            if importer_globals is None:
                importer_globals = caller.f_globals
            if importer_name is None:
                importer_name = caller.f_globals.get("__name__")
            importer_filename = caller.f_code.co_filename
        return importer_name, importer_globals, _normalize_origin(importer_filename)

    def _module_matches_context(
        module_name: str,
        importer_globals: dict | None,
        importer_filename: str | None,
    ) -> bool:
        if importer_globals is None or importer_filename is None:
            return False
        module = sys.modules.get(module_name)
        if not isinstance(module, ModuleType):
            return False
        if module.__dict__ is not importer_globals:
            return False
        return _module_origin(module) == importer_filename

    def _is_verified_importer(
        importer_name: str | None,
        importer_globals: dict | None,
        importer_filename: str | None,
    ) -> bool:
        if importer_name is None:
            return False
        trusted_context = trusted_modules.get(importer_name)
        if trusted_context is not None:
            globals_id, origin = trusted_context
            return importer_globals is not None and id(importer_globals) == globals_id and importer_filename == origin
        if not _is_transitive_allowed_import(importer_name):
            return False
        return _module_matches_context(importer_name, importer_globals, importer_filename)

    def _record_trusted_module(module_name: str) -> None:
        module = sys.modules.get(module_name)
        if not isinstance(module, ModuleType):
            return
        origin = _module_origin(module)
        if origin is None:
            return
        trusted_modules[module_name] = (id(module.__dict__), origin)

    def _record_trusted_imports(name: str, fromlist: Sequence[str] | None, result: object) -> None:
        names = {name}
        if isinstance(result, ModuleType):
            names.add(result.__name__)
        for item in fromlist or ():
            if not isinstance(item, str) or item == "*":
                continue
            names.add(f"{name}.{item}")
            if isinstance(result, ModuleType):
                attr = getattr(result, item, None)
                if isinstance(attr, ModuleType):
                    names.add(attr.__name__)
        for module_name in names:
            _record_trusted_module(module_name)

    def _validate_runtime_import(
        name: str,
        importer_name: str | None,
        importer_globals: dict | None,
        importer_filename: str | None,
    ) -> None:
        nonlocal trusted_depth
        import_root = name.split(".", 1)[0]
        trusted_request = trusted_depth > 0 or _is_verified_importer(
            importer_name,
            importer_globals,
            importer_filename,
        )
        if trusted_request and (
            import_root in COMPILER_TRANSITIVE_ROOTS
            or import_root.startswith("_remote_module_")
        ):
            return
        _validate_import_name(name, local_modules, Path("<runtime>"))

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        nonlocal trusted_depth
        importer_name = None
        importer_globals = None
        importer_filename = None
        trusted_import = trusted_depth > 0
        result = None
        if level == 0:
            importer_name, importer_globals, importer_filename = _resolve_importer_context(globals)
            _validate_runtime_import(name, importer_name, importer_globals, importer_filename)
            trusted_import = trusted_depth > 0 or _is_verified_importer(
                importer_name,
                importer_globals,
                importer_filename,
            )
        if trusted_import:
            trusted_depth += 1
        try:
            result = original_import(name, globals, locals, fromlist, level)
            return result
        finally:
            if trusted_import:
                if result is not None:
                    _record_trusted_imports(name, fromlist, result)
                trusted_depth -= 1

    def guarded_import_module(name: str, package: str | None = None):
        nonlocal trusted_depth
        trusted_import = trusted_depth > 0
        result = None
        if not name.startswith("."):
            importer_name, importer_globals, importer_filename = _resolve_importer_context(None)
            _validate_runtime_import(name, importer_name, importer_globals, importer_filename)
            trusted_import = trusted_depth > 0 or _is_verified_importer(
                importer_name,
                importer_globals,
                importer_filename,
            )
        if trusted_import:
            trusted_depth += 1
        try:
            result = original_import_module(name, package)
            return result
        finally:
            if trusted_import:
                if result is not None:
                    _record_trusted_imports(name, (), result)
                trusted_depth -= 1

    builtins.__import__ = guarded_import
    importlib.import_module = guarded_import_module
    try:
        yield
    finally:
        builtins.__import__ = original_import
        importlib.import_module = original_import_module


def _is_transitive_allowed_import(importer_name: str) -> bool:
    return importer_name == "torch" or importer_name.startswith("torch.") or importer_name == "triton" or importer_name.startswith("triton.")


def _wrap_submission_callable(callable_obj: Callable, local_modules: Set[str]) -> Callable:
    @wraps(callable_obj)
    def wrapped(*args, **kwargs):
        with _submission_import_guard(local_modules):
            return callable_obj(*args, **kwargs)

    return wrapped


def find_matching_variant(case: CaseSpec, variants: Sequence[VariantSpec]) -> VariantSpec:
    matches = [variant for variant in variants if variant.matches(case)]
    if not matches:
        raise ValueError(
            f"No declared variant matches case {case.case_id!r}. "
            "Add a catch-all variant or expand the manifest coverage."
        )
    if len(matches) > 1:
        names = ", ".join(variant.name for variant in matches)
        raise ValueError(
            f"Multiple declared variants match case {case.case_id!r}: {names}. "
            "Variant manifest rules must be unambiguous."
        )
    return matches[0]


def select_variant_representatives(
    suite: Sequence[CaseSpec],
    variants: Sequence[VariantSpec],
) -> list[tuple[VariantSpec, CaseSpec]]:
    representatives: list[tuple[VariantSpec, CaseSpec]] = []
    covered = set()
    for case in suite:
        variant = find_matching_variant(case, variants)
        if variant.name in covered:
            continue
        representatives.append((variant, case))
        covered.add(variant.name)
    return representatives


def _load_variant_manifest(module: ModuleType) -> tuple[VariantSpec, ...]:
    payload = getattr(module, "VARIANT_MANIFEST", None)
    if payload is None:
        raise AttributeError(
            "submission.py must define VARIANT_MANIFEST with at least one declared variant."
        )
    if not isinstance(payload, (list, tuple)):
        raise TypeError("VARIANT_MANIFEST must be a list or tuple.")
    if not payload:
        raise ValueError("VARIANT_MANIFEST must declare at least one variant.")
    if len(payload) > MAX_VARIANT_COUNT:
        raise ValueError(
            f"VARIANT_MANIFEST is too large: {len(payload)} > {MAX_VARIANT_COUNT}"
        )

    variants = tuple(_parse_variant_entry(item) for item in payload)
    names = [variant.name for variant in variants]
    if len(names) != len(set(names)):
        raise ValueError("VARIANT_MANIFEST variant names must be unique.")
    return variants


def _parse_variant_entry(item: object) -> VariantSpec:
    if isinstance(item, str):
        return VariantSpec(name=item)
    if not isinstance(item, dict):
        raise TypeError("Each VARIANT_MANIFEST entry must be a string name or an object.")

    name = item.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Each VARIANT_MANIFEST object must have a non-empty string 'name'.")

    families_payload = item.get("families", ())
    if not isinstance(families_payload, (list, tuple)):
        raise TypeError("Variant 'families' must be a list or tuple when provided.")
    families = tuple(_parse_family(value) for value in families_payload)

    min_t_max, max_t_max = _parse_numeric_range(item, "t_max", 0, 1 << 30, int)
    min_batch_heads, max_batch_heads = _parse_numeric_range(item, "batch_heads", 0, 1 << 30, int)
    min_batch_size, max_batch_size = _parse_numeric_range(item, "batch_size", 0, 1 << 30, int)
    min_num_heads, max_num_heads = _parse_numeric_range(item, "num_heads", 0, 1 << 30, int)
    min_window_blocks, max_window_blocks = _parse_numeric_range(item, "window_blocks", 0, 1 << 30, int)
    min_global_blocks, max_global_blocks = _parse_numeric_range(item, "global_blocks", 0, 1 << 30, int)
    min_retrieval_blocks, max_retrieval_blocks = _parse_numeric_range(item, "retrieval_blocks", 0, 1 << 30, int)
    min_retrieval_local_bias, max_retrieval_local_bias = _parse_numeric_range(
        item,
        "retrieval_local_bias",
        0.0,
        1.0,
        float,
    )

    return VariantSpec(
        name=name,
        families=families,
        min_t_max=min_t_max,
        max_t_max=max_t_max,
        min_batch_heads=min_batch_heads,
        max_batch_heads=max_batch_heads,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        min_num_heads=min_num_heads,
        max_num_heads=max_num_heads,
        min_window_blocks=min_window_blocks,
        max_window_blocks=max_window_blocks,
        min_global_blocks=min_global_blocks,
        max_global_blocks=max_global_blocks,
        min_retrieval_blocks=min_retrieval_blocks,
        max_retrieval_blocks=max_retrieval_blocks,
        min_retrieval_local_bias=min_retrieval_local_bias,
        max_retrieval_local_bias=max_retrieval_local_bias,
    )


def _parse_family(value: object) -> FamilyName:
    if value not in {"sliding_window", "sliding_window_global", "sliding_window_retrieval"}:
        raise ValueError(f"Unsupported family in VARIANT_MANIFEST: {value!r}")
    return value


def _parse_numeric_range(
    payload: dict,
    field_name: str,
    default_min: int | float,
    default_max: int | float,
    caster: Callable[[object], int | float],
) -> tuple[int | float, int | float]:
    exact_key = field_name
    min_key = f"min_{field_name}"
    max_key = f"max_{field_name}"

    has_exact = exact_key in payload
    has_min = min_key in payload
    has_max = max_key in payload

    if has_exact and (has_min or has_max):
        raise ValueError(
            f"Variant field {field_name!r} may use either {exact_key!r} or min/max bounds, not both."
        )

    if has_exact:
        exact_value = caster(payload[exact_key])
        return exact_value, exact_value

    min_value = caster(payload.get(min_key, default_min))
    max_value = caster(payload.get(max_key, default_max))
    if min_value > max_value:
        raise ValueError(
            f"Variant field {field_name!r} has invalid bounds: {min_key}={min_value} > {max_key}={max_value}."
        )
    return min_value, max_value
