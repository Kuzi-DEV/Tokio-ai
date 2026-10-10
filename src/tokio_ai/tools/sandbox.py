"""Run model-written signal code in an isolated process.

Layers, outermost first:

1. **A separate process** started with `python -I -S`, an empty environment
   (so API keys and tokens in yours aren't visible to it) and a working
   directory of its own. On Windows it is put in a job object that caps its
   memory and forbids it from starting any other process; on POSIX,
   resource limits do the same. It is killed after `timeout` seconds.
2. **An audit-hook allowlist** inside the child (PEP 578). After numpy and
   pandas are loaded and warmed up, the only audit events allowed are the two
   that pure computation raises (`builtins.id`, `sys._getframe`); any other
   event -- opening a file, a socket, starting a process, loading native code,
   importing a module, compiling code -- ends the process immediately.
3. **Data in, numbers out.** The child receives the price data on stdin and
   returns positions as base64 float64 in JSON; nothing it sends back is ever
   unpickled or executed by the parent.
4. **The code check** (`strategy.compile_signal`) runs first, in both
   processes, to turn obvious mistakes into readable errors early.

Audit hooks are not a guarantee on their own (PEP 578 says as much), which is
why the process boundary, the empty environment and the OS limits sit around
them.
"""

from __future__ import annotations

import base64
import json
import os
import pickle
import subprocess
import sys
import tempfile
from typing import Any

TIMEOUT_S = 60
MEMORY_LIMIT_BYTES = 2 * 1024 ** 3
MAX_OUTPUT_BYTES = 256 * 1024 ** 2

_BOOT = (
    "import sys,pickle\n"
    "job=pickle.load(sys.stdin.buffer)\n"
    "sys.path[:]=job['path']\n"
    "from tokio_ai.tools._sandbox_child import main\n"
    "main(job)\n"
)


class SandboxError(RuntimeError):
    """The isolated run failed: blocked action, timeout, memory, or the signal's own error."""


def run_isolated(code: str, name: str, data, requests: list[tuple[dict, int]],
                 timeout: float = TIMEOUT_S) -> list:
    """Positions for each (params, length) request, computed in an isolated child.

    Returns a list with a float64 numpy array per request, or a string with
    the signal's error message for that request.
    """
    import numpy as np

    job = {
        "path": list(sys.path),
        "code": code,
        "name": name,
        "index": data.index.asi8.tolist() if hasattr(data.index, "asi8") else list(range(len(data))),
        "columns": {c: data[c].to_numpy(dtype=float).tolist() for c in data.columns},
        "requests": [(dict(p), int(n)) for p, n in requests],
    }
    env = {}
    if os.name == "nt":  # the interpreter and numpy need these to start on Windows
        for k in ("SYSTEMROOT", "WINDIR"):
            if os.environ.get(k):
                env[k] = os.environ[k]
    with tempfile.TemporaryDirectory(prefix="tokio_sandbox_") as cwd:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", _BOOT],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, cwd=cwd, preexec_fn=_posix_limits if os.name != "nt" else None,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        job_handle = _windows_job(proc) if os.name == "nt" else None
        try:
            out, err = proc.communicate(pickle.dumps(job), timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise SandboxError(f"{name} ran longer than {timeout:g}s and was stopped")
        finally:
            if job_handle is not None:
                _close_handle(job_handle)
    stderr = err.decode(errors="replace")
    if "TOKIO_SANDBOX_BLOCKED" in stderr:
        event = stderr.split("TOKIO_SANDBOX_BLOCKED", 1)[1].split()[0]
        raise SandboxError(f"{name} tried to do something other than compute a signal "
                           f"({event}); the sandbox stopped it")
    if proc.returncode != 0:
        last = stderr.strip().splitlines()[-1] if stderr.strip() else f"exit code {proc.returncode}"
        raise SandboxError(f"{name} failed in the sandbox: {last[:400]}")
    if len(out) > MAX_OUTPUT_BYTES:
        raise SandboxError(f"{name} produced too much output")
    try:
        results = json.loads(out.decode())
    except ValueError as e:
        raise SandboxError(f"{name}: unreadable result from the sandbox") from e
    if not isinstance(results, list) or len(results) != len(requests):
        raise SandboxError(f"{name}: wrong number of results from the sandbox")
    parsed = []
    for (params, n), r in zip(requests, results):
        if isinstance(r, dict) and isinstance(r.get("b64"), str):
            arr = np.frombuffer(base64.b64decode(r["b64"]), dtype="<f8").astype(float)
            if arr.shape != (n,):
                raise SandboxError(f"{name}: result of the wrong length")
            parsed.append(arr)
        elif isinstance(r, dict) and isinstance(r.get("error"), str):
            parsed.append(r["error"][:400])
        else:
            raise SandboxError(f"{name}: malformed result from the sandbox")
    return parsed


class IsolatedSignal:
    """A signal whose code only ever runs in the sandbox; usable anywhere `Lab` takes a signal.

    `Lab` calls `prefetch` with every (params, length) it is about to need
    (the full run plus its lookahead cuts), so one child process serves a
    whole sweep; `__call__` then answers from that cache.
    """

    def __init__(self, code: str, name: str, timeout: float = TIMEOUT_S):
        from .strategy import compile_signal

        compile_signal(code, name)  # fail fast in the parent on obvious problems; result unused
        self.code, self.__name__, self.timeout = code, name, timeout
        self._cache: dict = {}
        self._data_key = None

    def _key(self, params, n):
        return (n, tuple(sorted(params.items())))

    def prefetch(self, data, requests: list[tuple[dict, int]]) -> None:
        key = (len(data), data.index[0], data.index[-1])
        if key != self._data_key:
            self._cache, self._data_key = {}, key
        todo = [(p, n) for p, n in requests if self._key(p, n) not in self._cache]
        if not todo:
            return
        for (p, n), res in zip(todo, run_isolated(self.code, self.__name__, data, todo, self.timeout)):
            self._cache[self._key(p, n)] = res

    def __call__(self, d, **params):
        import pandas as pd

        k = self._key(params, len(d))
        if k not in self._cache:
            self.prefetch(d, [(params, len(d))])
        res = self._cache[k]
        if isinstance(res, str):
            raise SandboxError(f"{self.__name__} raised {res}")
        return pd.Series(res, index=d.index)


# --------------------------------------------------------------------- OS limits

def _posix_limits():  # pragma: no cover - runs in the child on POSIX only
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_S + 5, TIMEOUT_S + 5))
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    except (ValueError, OSError):
        pass


def _windows_job(proc) -> Any:
    """Put the child in a job object: memory cap, no child processes, killed with the job."""
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x8
    JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x100
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    JobObjectExtendedLimitInformation = 9

    k32.CreateJobObjectW.restype = wintypes.HANDLE
    job = k32.CreateJobObjectW(None, None)
    if not job:
        proc.kill()
        raise SandboxError("could not create a job object for the sandbox")
    info = EXTENDED()
    info.BasicLimitInformation.LimitFlags = (JOB_OBJECT_LIMIT_ACTIVE_PROCESS | JOB_OBJECT_LIMIT_PROCESS_MEMORY
                                             | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
    info.BasicLimitInformation.ActiveProcessLimit = 1
    info.ProcessMemoryLimit = MEMORY_LIMIT_BYTES
    ok = k32.SetInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(info),
                                     ctypes.sizeof(info))
    ok = ok and k32.AssignProcessToJobObject(job, wintypes.HANDLE(int(proc._handle)))
    if not ok:
        proc.kill()
        _close_handle(job)
        raise SandboxError("could not apply the sandbox's process limits")
    return job


def _close_handle(h) -> None:
    import ctypes

    ctypes.WinDLL("kernel32").CloseHandle(h)
