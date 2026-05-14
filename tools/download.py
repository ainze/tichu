#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "tqdm",
# ]
# ///

import asyncio
import logging
from pathlib import Path

import httpx
from tqdm.asyncio import tqdm

BASE_URL = "http://tichulog.brettspielwelt.de/{}.tch"
OUTPUT_DIR = Path("/Users/andriesinze/workbench/tch-files")
FAILED_LOG = Path("failed.log")
START = 1
END = 2417589
WORKERS = 10
RETRIES = 3
TIMEOUT = 30

logging.basicConfig(level=logging.WARNING)


async def download_one(client: httpx.AsyncClient, file_id: int, sem: asyncio.Semaphore) -> int | None:
    dest = OUTPUT_DIR / f"{file_id}.tch"
    if dest.exists():
        return None  # already have it

    url = BASE_URL.format(file_id)
    async with sem:
        for attempt in range(RETRIES):
            try:
                r = await client.get(url, timeout=TIMEOUT)
                if r.status_code == 200:
                    dest.write_bytes(r.content)
                    return None
                if r.status_code == 404:
                    return file_id  # missing on server, no point retrying
                # unexpected status — retry
            except (httpx.RequestError, httpx.TimeoutException):
                pass
            if attempt < RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
        return file_id  # exhausted retries


async def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    ids = range(START, END + 1)
    sem = asyncio.Semaphore(WORKERS)
    failed: list[int] = []

    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = [download_one(client, i, sem) for i in ids]
        for coro in tqdm.as_completed(tasks, total=len(tasks), unit="file"):
            result = await coro
            if result is not None:
                failed.append(result)

    if failed:
        FAILED_LOG.write_text("\n".join(map(str, failed)) + "\n")
        print(f"\n{len(failed)} failures logged to {FAILED_LOG}")
    else:
        print("\nAll done, no failures.")


if __name__ == "__main__":
    asyncio.run(main())
