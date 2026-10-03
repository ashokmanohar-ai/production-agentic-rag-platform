import pytest

from app.runtime import DependencyStatus, RuntimeDiagnostics


@pytest.mark.asyncio
async def test_runtime_diagnostics_reports_ready(monkeypatch) -> None:
    diagnostics = object.__new__(RuntimeDiagnostics)

    async def ok():
        return DependencyStatus("ok")

    monkeypatch.setattr(diagnostics, "_postgres", ok)
    monkeypatch.setattr(diagnostics, "_redis", ok)
    monkeypatch.setattr(diagnostics, "_opensearch", ok)
    monkeypatch.setattr(diagnostics, "_ollama", ok)

    result = await diagnostics.check()
    assert result["status"] == "ready"
    assert all(item["status"] == "ok" for item in result["dependencies"].values())


@pytest.mark.asyncio
async def test_runtime_diagnostics_reports_not_ready(monkeypatch) -> None:
    diagnostics = object.__new__(RuntimeDiagnostics)

    async def ok():
        return DependencyStatus("ok")

    async def failed():
        return DependencyStatus("error", "missing_models:test")

    monkeypatch.setattr(diagnostics, "_postgres", ok)
    monkeypatch.setattr(diagnostics, "_redis", ok)
    monkeypatch.setattr(diagnostics, "_opensearch", ok)
    monkeypatch.setattr(diagnostics, "_ollama", failed)

    result = await diagnostics.check()
    assert result["status"] == "not_ready"
    assert result["dependencies"]["ollama"]["status"] == "error"
