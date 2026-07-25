"""Compatibility shim for the old exploratory import path.

Use `from bci_analysis.behavior import *` in new notebooks.
"""

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_PATH = _PROJECT_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from bci_analysis.behavior import *  # noqa: F403
