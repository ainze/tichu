"""`serve_inference` CLI — launch the FastAPI inference service.

`main(argv)` parses the YAML config, builds the app, and runs uvicorn.
A `--no-serve` flag short-circuits before uvicorn binds a port so the
integration tests can drive the app via `TestClient` instead.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
from fastapi import FastAPI

from tichu_inference.app import create_app


log = logging.getLogger("serve_inference")


def build_app_for_config(config: dict) -> FastAPI:
    return create_app(config)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Serve the Tichu inference service.")
    p.add_argument("--config", required=True, metavar="FILE", help="YAML config file")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=None,
                   help="Override config-supplied port.")
    p.add_argument("--no-serve", action="store_true",
                   help="Build the app and exit 0 without binding a port.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"config not found: {config_path}", file=sys.stderr)
        return 2
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    app = build_app_for_config(config)
    if args.no_serve:
        log.info("--no-serve: built app for %d agents and exiting",
                 len(app.state.agent_registry))
        return 0

    port = args.port or int(config.get("port", 8000))
    import uvicorn  # imported lazily so tests don't pull it in.
    uvicorn.run(app, host=args.host, port=port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
