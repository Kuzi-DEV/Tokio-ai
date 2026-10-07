"""The provenance stamp has to describe the code that actually ran."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError

import tokio_ai
from tokio_ai.rigor import provenance


def test_stamp_reports_the_running_source_version(monkeypatch):
    """Source ahead of the installed dist is the normal development state.

    Reporting the installed version there stamps a result with a version of
    the code that did not produce it -- observed live, where a 0.3.0 working
    tree stamped its verdicts "tokio-ai 0.2.0" because that is what was
    installed.
    """
    monkeypatch.setattr(tokio_ai, "__version__", "9.9.9")
    monkeypatch.setattr(provenance, "version", lambda name: "0.0.1" if name == "tokio-ai" else "1.2.3")
    text = provenance.stamp()
    assert "9.9.9" in text
    assert "0.0.1 installed" in text


def test_stamp_stays_terse_when_versions_agree(monkeypatch):
    monkeypatch.setattr(tokio_ai, "__version__", "1.0.0")
    monkeypatch.setattr(provenance, "version", lambda name: "1.0.0" if name == "tokio-ai" else "1.2.3")
    text = provenance.stamp()
    assert "tokio-ai 1.0.0 |" in text
    assert "installed" not in text


def test_stamp_works_when_not_installed_at_all(monkeypatch):
    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(tokio_ai, "__version__", "1.0.0")
    monkeypatch.setattr(provenance, "version", missing)
    text = provenance.stamp(agent=True)
    assert "tokio-ai 1.0.0" in text
    assert "openai unknown" in text


def test_library_results_do_not_mention_the_agent_backend(monkeypatch):
    monkeypatch.setattr(provenance, "version", lambda name: "1.2.3")
    assert "openai" not in provenance.stamp()
    assert "openai 1.2.3" in provenance.stamp(agent=True)


def test_stamp_includes_python_version():
    import platform

    assert platform.python_version() in provenance.stamp()
