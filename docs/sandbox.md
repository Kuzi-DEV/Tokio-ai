# Sandbox for model-written strategies

**Status: awaiting independent security review. Not enabled in any release.**

The agent's `backtest_strategy` tool turns a plain-English description into
Python (`def signal(d, **params)`) and runs it. That code is written by a
language model, which may misunderstand or be influenced by text it has read
(a filing, a web page), so it is treated as untrusted.

## Goal

The strategy code may compute positions from the price data it's given, and
nothing else: no reading or writing files, no network, no starting processes,
no access to the user's environment (API keys), no loading native code, and
no way to affect the parent process except through the numbers it returns.

## Layers

| layer | where | what it does |
|---|---|---|
| Code check | `tools/strategy.py` `compile_signal` | Rejects imports, dunder names, and a list of known file/IO attributes before anything runs. An early filter for readable errors, **not** a security boundary. |
| Separate process | `tools/sandbox.py` `run_isolated` | `python -I -S`, an empty environment (only `SYSTEMROOT`/`WINDIR` on Windows), a fresh temporary working directory, killed after a timeout. |
| OS limits | `sandbox.py` `_windows_job` / `_posix_limits` | Windows job object: 2 GB memory cap, at most 1 active process (no children), killed when the job closes. POSIX: address-space, CPU and process-count limits. |
| Audit-hook allowlist | `tools/_sandbox_child.py` | After numpy/pandas are imported and common operations warmed up, a PEP 578 hook allows only `builtins.id` and `sys._getframe` (measured as the only events ordinary signal math raises) and exits the process on any other event. |
| Data in, numbers out | both | The parent sends a pickle (parent to child only); the child returns positions as base64 float64 inside JSON. The parent never unpickles or executes anything the child sends. |

## What to review

1. Can strategy code, starting from what it is given (`d`, `pd`, `np`, the
   allowed builtins), perform any action from the Goal list?
2. Does any action that matters avoid raising an audit event that the hook
   sees (native extensions, already-imported modules' internals, the
   warm-up order)?
3. Is the job object applied before the child can do anything? (The child
   blocks reading stdin until the parent has written the job, which happens
   after `AssignProcessToJobObject`.)
4. Can the child's output influence the parent beyond the returned numbers
   (oversized output, malformed JSON, timing)?
5. On Linux/macOS: are the `resource` limits sufficient, and should seccomp
   or a namespace sandbox be added?

## Known limits

- PEP 578 states audit hooks are not a complete sandbox on their own, which
  is why the process boundary, empty environment and OS limits surround it.
- The OS layer on Windows does not block network access by itself; network
  calls are expected to be stopped by the audit hook.
- Infinite loops and memory use are bounded by the timeout and memory cap,
  not prevented.
