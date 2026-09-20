"""프로젝트 unittest 탐색과 Python 3.10 호환 설정."""
from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    import tomli

    sys.modules.setdefault("tomllib", tomli)
