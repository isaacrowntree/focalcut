"""focalcut — split video into focal-subject segments with SAM 3 object permanence."""
from .segment import Segment, segment
from .track import Sam3Tracker, Detection

__version__ = "0.1.0"
__all__ = ["Segment", "segment", "Sam3Tracker", "Detection"]
