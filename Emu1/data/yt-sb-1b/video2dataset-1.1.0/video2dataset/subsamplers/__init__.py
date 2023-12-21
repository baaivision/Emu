"""
Transform video and audio by reducing fps, extracting videos, changing resolution, reducing bitrate, etc.
"""

from .audio_rate_subsampler import AudioRateSubsampler
from .clipping_subsampler import ClippingSubsampler, get_seconds
from .frame_subsampler import FrameSubsampler
from .noop_subsampler import NoOpSubsampler
from .resolution_subsampler import ResolutionSubsampler
from .cut_detection_subsampler import CutDetectionSubsampler
