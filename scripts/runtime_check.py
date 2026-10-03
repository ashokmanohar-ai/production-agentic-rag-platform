from __future__ import annotations

import argparse
import asyncio
import json

from app.config import get_settings
from app.persistence.database import build_session_factory
from app.runtime import RuntimeDiagnostics


async def main() -> int:
    parser = argparse.ArgumentParser(description="Check runtime dependencies and Ollama models")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    result = await RuntimeDiagnostics(
        settings, build_session_factory(settings.database_url)
    ).check()
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"runtime: {result['status']}")
        for name, status in result["dependencies"].items():
            detail = f" ({status['detail']})" if status["detail"] else ""
            print(f"- {name}: {status['status']}{detail}")
    return 0 if result["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
