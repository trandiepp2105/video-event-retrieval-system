from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    module_root = Path(__file__).resolve().parents[3] / "subtitle_ocr_extraction_module"
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

    from main import main as legacy_main

    legacy_main()
