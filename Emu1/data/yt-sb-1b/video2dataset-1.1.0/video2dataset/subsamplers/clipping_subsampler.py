"""
clipping subsampler turns full videos into clips of videos according to clip_col

TODO: implement subtitle splitting (can be done just by indexing subtitle dict during clipping
"""
import os
import copy
import glob
import ffmpeg
import tempfile

from datetime import datetime


def get_seconds(t):
    if not isinstance(t, str):
        return float(t)  # already seconds
    time_format = "%H:%M:%S.%f"  # TODO: maybe paramaterize this?
    t_obj = datetime.strptime(t, time_format).time()
    return t_obj.second + t_obj.microsecond / 1e6 + t_obj.minute * 60 + t_obj.hour * 3600


class ClippingSubsampler:
    """
    Cuts videos up into segments according to the 'clips' metadata

    expects:
    - clips to be sorted in increasing order and non-overlapping
    - time to be in the format "%H:%M:%S.%f", or a number representing the second of the timestamp
    """

    def __init__(self, oom_clip_count, encode_formats):
        self.oom_clip_count = oom_clip_count
        self.encode_formats = encode_formats

    def __call__(self, streams, metadata):
        clips = metadata.pop("clips")
        lines = metadata.pop("lines") if "lines" in metadata else None

        if isinstance(clips[0], float):  # make sure clips looks like [[start, end]] and not [start, end]
            clips = [clips]

        ind = 2
        # we assume there's always one clip which we want to take

        s_p, e_p = clips[0]
        s_p, e_p = get_seconds(s_p), get_seconds(e_p)
        splits = [s_p, e_p]
        # list of indicies of clips to take, used to discard non-contiguous sections
        take_inds = [1]

        # TODO: make nicer
        for s, e in clips[1:]:
            s, e = get_seconds(s), get_seconds(e)

            if s - e_p <= 1.0:  # no one needs 1.0 second clips + creates less files
                splits += [e]
                take_inds.append(ind)
            else:
                splits += [s, e]
                take_inds.append(ind + 1)

            ind += 1 if s - e_p <= 1.0 else 2
            e_p = e
        segment_times = ",".join([str(spl) for spl in splits])

        streams_clips = {}

        for k in streams.keys():
            stream_bytes = streams[k]
            if stream_bytes is None:
                continue
            encode_format = self.encode_formats[k]

            with tempfile.TemporaryDirectory() as tmpdir:
                # TODO: we need to put the extension into the metadata
                # TODO: This can be done better using pipes I just don't feel like sinking too much time into this rn
                with open(os.path.join(tmpdir, f"input.{encode_format}"), "wb") as f:
                    f.write(stream_bytes)
                try:
                    _ = (
                        ffmpeg.input(f"{tmpdir}/input.{encode_format}")
                        .output(
                            f"{tmpdir}/clip_%d.{encode_format}",
                            c="copy",
                            map=0,
                            f="segment",
                            segment_times=segment_times,
                            reset_timestamps=1,
                        )
                        .run(capture_stdout=True, quiet=True)
                    )

                except Exception as err:  # pylint: disable=broad-except
                    return [], [], str(err)

                stream_clips = glob.glob(f"{tmpdir}/clip*.{encode_format}")
                stream_clips.sort()
                correct_clips = []
                for clip_id, (clip, ind) in enumerate(zip(clips, take_inds)):
                    if ind < len(stream_clips):
                        correct_clips.append((clip_id, clip, stream_clips[ind]))
                # clips_lost = len(take_inds) - len(correct_clips) # TODO report this somehow

                stream_clips, metadata_clips = [], []
                for i, (clip_id, clip_span, clip_pth) in enumerate(correct_clips):
                    with open(clip_pth, "rb") as vid_f:
                        clip_bytes = vid_f.read()
                    stream_clips.append(clip_bytes)

                    clip_key = "{clip_id:0{oom_clip_count}d}".format(  # pylint: disable=consider-using-f-string
                        clip_id=clip_id, oom_clip_count=self.oom_clip_count
                    )
                    meta_clip = copy.deepcopy(metadata)
                    # set the timeframe of this clip
                    meta_clip["clips"] = [clip_span]
                    meta_clip["key"] = f"{meta_clip['key']}_{clip_key}"
                    if lines is not None:
                        meta_clip["yt_meta_dict"]["subtitles"] = lines[i]
                    metadata_clips.append(meta_clip)

                streams_clips[k] = stream_clips

        return streams_clips, metadata_clips, None
