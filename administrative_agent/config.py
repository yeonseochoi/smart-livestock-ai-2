from __future__ import annotations

import os
from pathlib import Path


def load_env(path: Path) -> None:
    """외부 패키지 없이 단순 KEY=VALUE 형식의 .env를 읽는다."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)
