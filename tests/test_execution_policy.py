import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from attention_kernel_challenge.execution_policy import (
    PolicyViolationError,
    _ALLOWED_CTYPES_HANDLES,
    _AUDIT_GUARD_ACTIVE,
    _guarded_ctypes_loader,
    _guarded_posix_spawn_callable,
    _guarded_subprocess_callable,
    _is_allowed_ctypes_library,
    _is_allowed_subprocess_argv,
    _matches_triton_nvidia_driver_frame,
    _matches_triton_runtime_build_frame,
    _should_simulate_missing_subprocess,
    _submission_audit_hook,
    CompilationCacheMonitor,
    submission_runtime_guard,
)


class ExecutionPolicyTests(unittest.TestCase):
    def test_allowed_ctypes_library_recognizes_cuda_driver_names(self) -> None:
        self.assertTrue(_is_allowed_ctypes_library("libnvidia-ml.so.1"))
        self.assertTrue(_is_allowed_ctypes_library("/usr/lib/libcuda.so.1"))
        self.assertFalse(_is_allowed_ctypes_library("libc.so.6"))
        self.assertFalse(_is_allowed_ctypes_library(None))

    def test_guarded_ctypes_loader_rejects_disallowed_library(self) -> None:
        guard_token = _AUDIT_GUARD_ACTIVE.set(True)
        handles_token = _ALLOWED_CTYPES_HANDLES.set(set())
        try:
            loader = _guarded_ctypes_loader(lambda *_args, **_kwargs: None, "ctypes.CDLL")
            with self.assertRaises(PolicyViolationError):
                loader("libc.so.6")
        finally:
            _ALLOWED_CTYPES_HANDLES.reset(handles_token)
            _AUDIT_GUARD_ACTIVE.reset(guard_token)

    def test_guarded_ctypes_loader_records_allowed_handle(self) -> None:
        guard_token = _AUDIT_GUARD_ACTIVE.set(True)
        handles = set()
        handles_token = _ALLOWED_CTYPES_HANDLES.set(handles)
        try:
            loader = _guarded_ctypes_loader(lambda *_args, **_kwargs: SimpleNamespace(_handle=123), "ctypes.CDLL")
            library = loader("libnvidia-ml.so.1")
            self.assertEqual(library._handle, 123)
            self.assertEqual(handles, {123})
        finally:
            _ALLOWED_CTYPES_HANDLES.reset(handles_token)
            _AUDIT_GUARD_ACTIVE.reset(guard_token)

    def test_submission_audit_hook_blocks_untracked_dlsym(self) -> None:
        guard_token = _AUDIT_GUARD_ACTIVE.set(True)
        handles_token = _ALLOWED_CTYPES_HANDLES.set({123})
        try:
            with self.assertRaises(PolicyViolationError):
                _submission_audit_hook("ctypes.dlsym", (SimpleNamespace(_handle=456), "symbol"))
            _submission_audit_hook("ctypes.dlsym", (SimpleNamespace(_handle=123), "symbol"))
        finally:
            _ALLOWED_CTYPES_HANDLES.reset(handles_token)
            _AUDIT_GUARD_ACTIVE.reset(guard_token)

    def test_allowed_subprocess_argv_accepts_known_safe_commands(self) -> None:
        self.assertTrue(_is_allowed_subprocess_argv(["/sbin/ldconfig", "-p"]))
        self.assertTrue(_is_allowed_subprocess_argv(["uname", "-p"]))
        self.assertFalse(_is_allowed_subprocess_argv(["/bin/echo", "hi"]))

    def test_triton_frame_matchers_accept_pyc_paths(self) -> None:
        self.assertTrue(_matches_triton_runtime_build_frame(Path("runtime/__pycache__/build.cpython-311.pyc")))
        self.assertTrue(
            _matches_triton_nvidia_driver_frame(
                Path("backends/nvidia/__pycache__/driver.cpython-311.pyc")
            )
        )
        self.assertTrue(
            _matches_triton_nvidia_driver_frame(
                Path("third_party/nvidia/backend/__pycache__/driver.cpython-311.pyc")
            )
        )

    def test_guarded_subprocess_callable_rejects_disallowed_command(self) -> None:
        guard_token = _AUDIT_GUARD_ACTIVE.set(True)
        try:
            guarded = _guarded_subprocess_callable(lambda *_args, **_kwargs: None, "subprocess.check_output")
            with self.assertRaises(PolicyViolationError):
                guarded(["/bin/echo", "hi"])
        finally:
            _AUDIT_GUARD_ACTIVE.reset(guard_token)

    def test_guarded_subprocess_callable_allows_ldconfig_lookup(self) -> None:
        guard_token = _AUDIT_GUARD_ACTIVE.set(True)
        try:
            guarded = _guarded_subprocess_callable(lambda *_args, **_kwargs: "ok", "subprocess.check_output")
            self.assertEqual(guarded(["/sbin/ldconfig", "-p"]), "ok")
        finally:
            _AUDIT_GUARD_ACTIVE.reset(guard_token)

    def test_guarded_subprocess_callable_simulates_missing_safe_probe(self) -> None:
        guard_token = _AUDIT_GUARD_ACTIVE.set(True)
        try:
            guarded = _guarded_subprocess_callable(lambda *_args, **_kwargs: None, "subprocess.check_output")
            with self.assertRaises(FileNotFoundError):
                guarded(["nvcc", "--version"])
            with self.assertRaises(FileNotFoundError):
                guarded(["file", "-b", sys.executable])
        finally:
            _AUDIT_GUARD_ACTIVE.reset(guard_token)

    def test_allowed_subprocess_argv_accepts_trusted_triton_host_compiler(self) -> None:
        with patch(
            "attention_kernel_challenge.execution_policy._trusted_subprocess_roots",
            return_value=(Path("/trusted"),),
        ), patch(
            "attention_kernel_challenge.execution_policy._is_trusted_triton_launcher_build_context",
            return_value=True,
        ), patch.dict(
            os.environ,
            {
                "TMPDIR": "/sandbox/tmp",
                "TRITON_CACHE_DIR": "/sandbox/cache/triton",
            },
            clear=False,
        ):
            self.assertTrue(
                _is_allowed_subprocess_argv(
                    [
                        "/trusted/gcc",
                        "/sandbox/tmp/main.c",
                        "-O3",
                        "-shared",
                        "-fPIC",
                        "-o",
                        "/sandbox/tmp/triton_launcher.so",
                    ]
                )
            )

    def test_allowed_subprocess_argv_rejects_host_compiler_without_triton_context(self) -> None:
        with patch(
            "attention_kernel_challenge.execution_policy._trusted_subprocess_roots",
            return_value=(Path("/trusted"),),
        ), patch.dict(
            os.environ,
            {
                "TMPDIR": "/sandbox/tmp",
                "TRITON_CACHE_DIR": "/sandbox/cache/triton",
            },
            clear=False,
        ):
            self.assertFalse(
                _is_allowed_subprocess_argv(
                    [
                        "/trusted/gcc",
                        "/sandbox/tmp/main.c",
                        "-shared",
                        "-fPIC",
                        "-o",
                        "/sandbox/tmp/triton_launcher.so",
                    ]
                )
            )

    def test_allowed_subprocess_argv_rejects_host_compiler_writing_outside_cache_roots(self) -> None:
        with patch(
            "attention_kernel_challenge.execution_policy._trusted_subprocess_roots",
            return_value=(Path("/trusted"),),
        ), patch(
            "attention_kernel_challenge.execution_policy._is_trusted_triton_launcher_build_context",
            return_value=True,
        ), patch.dict(
            os.environ,
            {
                "TMPDIR": "/sandbox/tmp",
                "TRITON_CACHE_DIR": "/sandbox/cache/triton",
            },
            clear=False,
        ):
            self.assertFalse(
                _is_allowed_subprocess_argv(
                    [
                        "/trusted/gcc",
                        "/sandbox/tmp/main.c",
                        "-shared",
                        "-fPIC",
                        "-o",
                        "/tmp/triton_launcher.so",
                    ]
                )
            )

    def test_allowed_subprocess_argv_accepts_trusted_ptxas_path(self) -> None:
        with patch(
            "attention_kernel_challenge.execution_policy._trusted_subprocess_roots",
            return_value=(Path("/trusted"),),
        ):
            self.assertTrue(
                _is_allowed_subprocess_argv(
                    ["/trusted/ptxas", "-lineinfo", "-v", "--gpu-name=sm_90a", "/tmp/kernel.ptx", "-o", "/tmp/kernel.ptx.o"]
                )
            )
            self.assertFalse(
                _is_allowed_subprocess_argv(
                    ["/tmp/ptxas", "-lineinfo", "-v", "--gpu-name=sm_90a", "/tmp/kernel.ptx", "-o", "/tmp/kernel.ptx.o"]
                )
            )

    def test_allowed_subprocess_argv_accepts_trusted_compiler_symlink_path(self) -> None:
        with patch(
            "attention_kernel_challenge.execution_policy._trusted_subprocess_roots",
            return_value=(Path("/trusted"),),
        ), patch(
            "attention_kernel_challenge.execution_policy._is_trusted_triton_launcher_build_context",
            return_value=True,
        ), patch(
            "attention_kernel_challenge.execution_policy._resolve_subprocess_command",
            return_value=Path("/trusted/gcc-12"),
        ), patch.dict(
            os.environ,
            {
                "TMPDIR": "/sandbox/tmp",
                "TRITON_CACHE_DIR": "/sandbox/cache/triton",
            },
            clear=False,
        ):
            self.assertTrue(
                _is_allowed_subprocess_argv(
                    [
                        "/trusted/gcc",
                        "/sandbox/tmp/main.c",
                        "-shared",
                        "-fPIC",
                        "-o",
                        "/sandbox/tmp/triton_launcher.so",
                    ]
                )
            )

    def test_guarded_posix_spawn_callable_allows_trusted_ptxas(self) -> None:
        guard_token = _AUDIT_GUARD_ACTIVE.set(True)
        try:
            with patch(
                "attention_kernel_challenge.execution_policy._trusted_subprocess_roots",
                return_value=(Path("/trusted"),),
            ):
                guarded = _guarded_posix_spawn_callable(lambda *args, **kwargs: "ok", "os.posix_spawn")
                self.assertEqual(
                    guarded(
                        "/trusted/ptxas",
                        ["/trusted/ptxas", "-lineinfo", "-v", "--gpu-name=sm_90a", "/tmp/kernel.ptx", "-o", "/tmp/kernel.ptx.o"],
                        {},
                    ),
                    "ok",
                )
        finally:
            _AUDIT_GUARD_ACTIVE.reset(guard_token)

    def test_guarded_subprocess_callable_allows_trusted_triton_check_call(self) -> None:
        guard_token = _AUDIT_GUARD_ACTIVE.set(True)
        try:
            with patch(
                "attention_kernel_challenge.execution_policy._trusted_subprocess_roots",
                return_value=(Path("/trusted"),),
            ), patch(
                "attention_kernel_challenge.execution_policy._is_trusted_triton_launcher_build_context",
                return_value=True,
            ), patch.dict(
                os.environ,
                {
                    "TMPDIR": "/sandbox/tmp",
                    "TRITON_CACHE_DIR": "/sandbox/cache/triton",
                },
                clear=False,
            ):
                guarded = _guarded_subprocess_callable(lambda *_args, **_kwargs: "ok", "subprocess.check_call")
                self.assertEqual(
                    guarded(
                        [
                            "/trusted/gcc",
                            "/sandbox/tmp/main.c",
                            "-shared",
                            "-fPIC",
                            "-o",
                            "/sandbox/tmp/triton_launcher.so",
                        ]
                    ),
                    "ok",
                )
        finally:
            _AUDIT_GUARD_ACTIVE.reset(guard_token)

    def test_guarded_subprocess_callable_allows_trusted_triton_call(self) -> None:
        guard_token = _AUDIT_GUARD_ACTIVE.set(True)
        try:
            with patch(
                "attention_kernel_challenge.execution_policy._trusted_subprocess_roots",
                return_value=(Path("/trusted"),),
            ), patch(
                "attention_kernel_challenge.execution_policy._is_trusted_triton_launcher_build_context",
                return_value=True,
            ), patch.dict(
                os.environ,
                {
                    "TMPDIR": "/sandbox/tmp",
                    "TRITON_CACHE_DIR": "/sandbox/cache/triton",
                },
                clear=False,
            ):
                guarded = _guarded_subprocess_callable(lambda *_args, **_kwargs: 0, "subprocess.call")
                self.assertEqual(
                    guarded(
                        [
                            "/trusted/gcc",
                            "/sandbox/tmp/main.c",
                            "-shared",
                            "-fPIC",
                            "-o",
                            "/sandbox/tmp/triton_launcher.so",
                        ]
                    ),
                    0,
                )
        finally:
            _AUDIT_GUARD_ACTIVE.reset(guard_token)

    def test_submission_audit_hook_allows_trusted_posix_spawn(self) -> None:
        guard_token = _AUDIT_GUARD_ACTIVE.set(True)
        try:
            with patch(
                "attention_kernel_challenge.execution_policy._trusted_subprocess_roots",
                return_value=(Path("/trusted"),),
            ):
                _submission_audit_hook(
                    "os.posix_spawn",
                    ("/trusted/ptxas", ["/trusted/ptxas", "--version"], {}),
                )
        finally:
            _AUDIT_GUARD_ACTIVE.reset(guard_token)

    def test_safe_probe_detection_matches_expected_commands(self) -> None:
        self.assertTrue(_should_simulate_missing_subprocess(["nvcc", "--version"]))
        self.assertTrue(_should_simulate_missing_subprocess(["file", "-b", sys.executable]))
        self.assertFalse(_should_simulate_missing_subprocess(["/bin/echo", "hi"]))

    def test_compilation_cache_monitor_sets_single_thread_inductor(self) -> None:
        original = os.environ.get("TORCHINDUCTOR_COMPILE_THREADS")
        original_tmpdir = os.environ.get("TMPDIR")
        original_system_tmpdir = tempfile.gettempdir()
        with CompilationCacheMonitor():
            self.assertEqual(os.environ["TORCHINDUCTOR_COMPILE_THREADS"], "1")
            self.assertTrue(os.environ["TMPDIR"].startswith("/"))
            self.assertEqual(os.environ["TMPDIR"], os.environ["TMP"])
            self.assertEqual(os.environ["TMPDIR"], os.environ["TEMP"])
            self.assertEqual(tempfile.gettempdir(), os.environ["TMPDIR"])
        self.assertEqual(os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"), original)
        self.assertEqual(os.environ.get("TMPDIR"), original_tmpdir)
        tempfile.tempdir = None
        self.assertEqual(tempfile.gettempdir(), original_system_tmpdir)

    def test_submission_runtime_guard_still_blocks_pythonapi_lookup(self) -> None:
        import ctypes

        with submission_runtime_guard():
            with self.assertRaises(PolicyViolationError):
                ctypes.pythonapi.Py_IsInitialized


if __name__ == "__main__":
    unittest.main()
