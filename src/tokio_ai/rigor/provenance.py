"""What ran, on what, with what -- the minimum needed for someone else (or
future you) to reproduce a result rather than just trust it."""

from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version


def stamp(agent: bool = False) -> str:
    """Version stamp for a result; `agent=True` adds the openai client's version (the agent's backend)."""
    # The running module's own __version__ is authoritative, NOT the
    # installed distribution metadata. Those disagree whenever the source
    # tree is ahead of the last `pip install` -- which is the normal state
    # while developing, and exactly when a result is most likely to be
    # pasted somewhere. Reporting the installed version there would stamp a
    # result with a version of the code that did not produce it, which
    # defeats the only reason this function exists.
    from .. import __version__ as tokio_version

    try:
        installed = version("tokio-ai")
    except PackageNotFoundError:
        installed = None
    if installed is not None and installed != tokio_version:
        tokio_version = f"{tokio_version} (source; {installed} installed)"

    out = f"tokio-ai {tokio_version} | Python {platform.python_version()}"
    if agent:
        try:
            openai_version = version("openai")
        except PackageNotFoundError:
            openai_version = "unknown"
        out += f" | openai {openai_version}"
    return out
