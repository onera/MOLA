#!/usr/bin/env python3
"""Add a released documentation version to Sphinx's version switcher."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def update_versions(path: Path, version: str) -> None:
    entries = json.loads(path.read_text(encoding="utf-8"))
    entries = [entry for entry in entries if entry.get("version") != version]
    entry = {
        "name": version,
        "version": version,
        "url": f"https://numerics.gitlab-pages.onera.net/mola/{version}/",
    }
    insertion_index = next(
        (index for index, item in enumerate(entries) if item.get("version") == "Dev"),
        0,
    ) + 1
    entries.insert(insertion_index, entry)
    path.write_text(json.dumps(entries, indent=4) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: update_doc_versions.py <versions.json> <version>")
    update_versions(Path(sys.argv[1]), sys.argv[2])
