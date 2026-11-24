"""brickexp: experiments on internalizing physical intuition from brick building."""
import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "BrickGPT" / "src"
if _VENDOR.exists() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))
