#!/usr/bin/env python3
"""Fail on likely private paths, secrets, or accidentally committed large files."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".csv", ".ipynb", ".svg"}
PATTERNS = {
    "absolute home path": re.compile(r"/(?:home|Users)/[^/\s]+/"),
    "obsolete research path": re.compile(r"AVOSurprise|avo-structure-inversion"),
    "generic API key assignment": re.compile(r"(?i)(?:api[_-]?key|secret|token)\s*[:=]\s*['\"][^'\"]{8,}"),
    "private key": re.compile(r"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY"),
}
MAX_FILE_BYTES = 5 * 1024 * 1024
LOCAL_ONLY_FILES = {Path("configs/paths.yaml")}


def main() -> None:
    failures: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        relative = path.relative_to(ROOT)
        # This file is intentionally git-ignored and contains machine-local,
        # authorized data paths. Scan the committed paths.example.yaml instead.
        if relative in LOCAL_ONLY_FILES:
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            failures.append(f"large file: {relative} ({path.stat().st_size} bytes)")
        if path.suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            for label, pattern in PATTERNS.items():
                if pattern.search(text):
                    failures.append(f"{label}: {relative}")
    if failures:
        raise SystemExit("Public-repository scan failed:\n  " + "\n  ".join(sorted(set(failures))))
    print("Public-repository scan passed")


if __name__ == "__main__":
    main()
