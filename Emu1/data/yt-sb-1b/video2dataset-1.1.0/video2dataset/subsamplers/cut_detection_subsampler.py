"""
cut detection subsampler detects cuts in a video
"""
import numpy as np
from scenedetect import AdaptiveDetector, SceneManager, open_video
import os
import tempfile


def get_scenes_from_scene_manager(scene_manager, cut_detection_mode):
    """
    Returns a list of cuts from a scene manager given a cut detection mode
    """
    scene_list = scene_manager.get_scene_list(start_in_scene=True)
    scene = []

    for clip in scene_list:
        scene.append([clip[0].get_frames(), clip[1].get_frames()])

    if cut_detection_mode == "longest":  # we have multiple cuts, pick the longest
        longest_clip = np.argmax([clip[1] - clip[0] for clip in scene])
        scene = [scene[longest_clip]]

    return scene


class CutDetectionSubsampler:
    """
    Detects cuts in input videos and returns contiguous segments in a video as metadata.

    expects:
    - cut_detection_mode to be either "longest" to pick the longest cut or "all" to pick all cuts
    - framerates to be None (for original fps only) or a list of target framerates to detect cuts in
    """

    def __init__(self, cut_detection_mode="all", framerates=None):
        self.framerates = framerates
        self.cut_detection_mode = cut_detection_mode

    def __call__(self, streams):
        video_bytes = streams["video"]

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "input.mp4")
            with open(video_path, "wb") as f:
                f.write(video_bytes)

            video = open_video(video_path)

            detector = AdaptiveDetector()
            scene_manager = SceneManager()
            scene_manager.add_detector(detector)

            cuts = {}
            original_fps = video.frame_rate
            cuts["original_fps"] = original_fps

            scene_manager.detect_scenes(video=video)
            cuts["cuts_original_fps"] = get_scenes_from_scene_manager(scene_manager, self.cut_detection_mode)
            if self.framerates is not None:
                for target_fps in self.framerates:
                    video.reset()

                    scene_manager = SceneManager()
                    detector = AdaptiveDetector()
                    scene_manager.add_detector(detector)
                    frame_skip = max(
                        int(original_fps // target_fps) - 1, 0
                    )  # if we take 1 frame and skip N frames we're sampling 1/N+1 % of the video
                    # so if we desire to sample 1/N of the video, we need to subtract one when doing frame skipping

                    scene_manager.detect_scenes(video=video, frame_skip=frame_skip)
                    cuts[f"cuts_{target_fps}"] = get_scenes_from_scene_manager(scene_manager, self.cut_detection_mode)
                    scene_manager.clear()

        return cuts
