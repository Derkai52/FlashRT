from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flash_rt.piinfra.schema import ProfileDB


def save_db(db: ProfileDB, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(db.to_dict(), indent=2))
    return path


def load_db(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
