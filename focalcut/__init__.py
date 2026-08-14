"""focalcut — split video into focal-subject segments with SAM 3 object permanence."""
from .segment import Segment, segment
from .track import Detection, Sam3Tracker

__version__ = "0.1.0"
__all__ = ["Detection", "Sam3Tracker", "Segment", "segment"]
