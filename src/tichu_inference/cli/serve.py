"""`serve_inference` CLI — launch the FastAPI inference service.

`main(argv)` parses the YAML config, builds the app, and runs uvicorn.
A `--no-serve` flag short-circuits before uvicorn binds a port so the
integration tests can drive the app via `TestClient` instead.

`--require-v7` is the deploy-time switch for the v7 wire contract (ADR-0044).
It stays OFF until the client sends `public.rich_history`, because the codec
decodes a missing accumulator to zeros — so an out-of-date client degrades the
policy's input silently, with nothing in any metric to show it.
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
    p.add_argument("--tape-log", metavar="FILE", default=None,
                   help="Append a human-reviewable decision tape to FILE, one block "
                        "per served Decision: public-state context, hand, chosen "
                        "action, top-k ranked alternatives, plus a single-line "
                        "`replay:` of the verbatim /act request so a flagged blunder "
                        "can be re-fed to /act later. Off by default; for diagnosing "
                        "play while playing visually.")
    p.add_argument("--require-v7", action="store_true",
                   help="Reject /act and /call payloads whose public state omits "
                        "`rich_history` (the v7 Rich History Block, ADR-0044). Off "
                        "by default so a pre-v7 client keeps working — but a missing "
                        "block decodes to 233 ZEROS that training saw populated, "
                        "which degrades play with nothing in any metric to show it. "
                        "Turn this on as soon as the client sends the block. Can "
                        "also be set as `require_v7: true` in the config.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"config not found: {config_path}", file=sys.stderr)
        return 2
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if args.tape_log:
        config["tape_log"] = args.tape_log
        log.info("decision tape -> %s", args.tape_log)
    # The flag can only turn strict mode ON, never off: a deployment that has
    # pinned `require_v7: true` in its config has an updated client, and letting
    # a stray argv silently re-enable the degraded path is the failure this
    # switch exists to prevent.
    if args.require_v7:
        config["require_v7"] = True

    app = build_app_for_config(config)
    if args.no_serve:
        log.info("--no-serve: built app for %d agents and exiting",
                 len(app.state.agent_registry))
        return 0

    port = args.port or int(config.get("port", 8000))
    import uvicorn  # imported lazily so tests don't pull it in.
    # access_log=False: the app's timing middleware emits the access line with
    # request duration (ms); uvicorn's default line has no timing.
    uvicorn.run(app, host=args.host, port=port, access_log=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
