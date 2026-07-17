#!/usr/bin/env python3
"""Fetch the pinned, build-time-only resources for the embedded local parser.

This script is run explicitly by developers and release CI. The shipped app
never downloads model files and contains no networking dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import os
from pathlib import Path
import ssl
import sys
import time
import urllib.error
import urllib.request


CHUNK_SIZE = 8 * 1024 * 1024
USER_AGENT = "log-parser-build/0.2.1"
NETWORK_TIMEOUT_SECONDS = 60
MAX_DOWNLOAD_ATTEMPTS = 4
INITIAL_RETRY_DELAY_SECONDS = 2

RESOURCES = (
    {
        "label": "Qwen2.5-1.5B-Instruct Q4_K_M model",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "url": (
            "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/"
            "91cad51170dc346986eccefdc2dd33a9da36ead9/"
            "qwen2.5-1.5b-instruct-q4_k_m.gguf"
        ),
        "sha256": "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e",
    },
    {
        "label": "Qwen2.5-1.5B-Instruct tokenizer",
        "filename": "qwen2.5-1.5b-instruct-tokenizer.json",
        "url": (
            "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/"
            "989aa7980e4cf806f80c7fef2b1adb7bc71aa306/tokenizer.json"
        ),
        "sha256": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    },
    {
        "label": "all-MiniLM-L6-v2 semantic-search model",
        "filename": "all-minilm-l6-v2-model.safetensors",
        "url": (
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/"
            "1110a243fdf4706b3f48f1d95db1a4f5529b4d41/model.safetensors"
        ),
        "sha256": "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
    },
    {
        "label": "all-MiniLM-L6-v2 tokenizer",
        "filename": "all-minilm-l6-v2-tokenizer.json",
        "url": (
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/"
            "1110a243fdf4706b3f48f1d95db1a4f5529b4d41/tokenizer.json"
        ),
        "sha256": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    },
    {
        "label": "all-MiniLM-L6-v2 configuration",
        "filename": "all-minilm-l6-v2-config.json",
        "url": (
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/"
            "1110a243fdf4706b3f48f1d95db1a4f5529b4d41/config.json"
        ),
        "sha256": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
    },
)


class ChecksumMismatchError(RuntimeError):
    """A completed download did not match its pinned digest."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def download(resource: dict[str, str], destination: Path) -> None:
    expected = resource["sha256"]
    if destination.is_file():
        actual = sha256_file(destination)
        if actual == expected:
            print(f"verified {resource['label']}: {destination}")
            return
        print(
            f"replacing checksum-mismatched {destination} "
            f"(expected {expected}, got {actual})",
            file=sys.stderr,
        )

    temporary = destination.with_name(destination.name + ".part")
    retryable_errors = (
        urllib.error.URLError,
        TimeoutError,
        ConnectionError,
        http.client.HTTPException,
        ssl.SSLError,
        ChecksumMismatchError,
    )

    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        temporary.unlink(missing_ok=True)
        request = urllib.request.Request(
            resource["url"], headers={"User-Agent": USER_AGENT}
        )
        print(
            f"downloading {resource['label']} -> {destination} "
            f"(attempt {attempt}/{MAX_DOWNLOAD_ATTEMPTS})"
        )
        try:
            with urllib.request.urlopen(
                request, timeout=NETWORK_TIMEOUT_SECONDS
            ) as response, temporary.open("wb") as output:
                digest = hashlib.sha256()
                while chunk := response.read(CHUNK_SIZE):
                    output.write(chunk)
                    digest.update(chunk)
            actual = digest.hexdigest()
            if actual != expected:
                raise ChecksumMismatchError(
                    f"checksum mismatch for {resource['label']}: "
                    f"expected {expected}, got {actual}"
                )
            os.replace(temporary, destination)
            print(f"verified {resource['label']}: {destination}")
            return
        except retryable_errors as error:
            if attempt == MAX_DOWNLOAD_ATTEMPTS:
                raise RuntimeError(
                    f"failed to download {resource['label']} after "
                    f"{MAX_DOWNLOAD_ATTEMPTS} attempts: {error}"
                ) from error
            delay = INITIAL_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
            print(
                f"download attempt {attempt} failed for {resource['label']}: "
                f"{error}; retrying in {delay} seconds",
                file=sys.stderr,
            )
        finally:
            temporary.unlink(missing_ok=True)

        time.sleep(delay)

    raise AssertionError("download retry loop ended unexpectedly")


def parse_args() -> argparse.Namespace:
    default_destination = (
        Path(__file__).resolve().parents[1] / "src-tauri" / "resources" / "models"
    )
    parser = argparse.ArgumentParser(
        description="Fetch checksum-pinned resources for offline AI search."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=default_destination,
        help=f"resource directory (default: {default_destination})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for resource in RESOURCES:
        download(resource, destination / resource["filename"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
