"""classes and functions for downloading videos"""
import os
import uuid
import requests
import yt_dlp
import io
import webvtt
import ffmpeg
from thumbframes_dl import YouTubeFrames

def video2audio(video, audio_format, tmp_dir):
    """extract audio from video"""
    path = f"{tmp_dir}/{str(uuid.uuid4())}.{audio_format}"
    num_streams = len(ffmpeg.probe(video)["streams"])
    ffmpeg_args = {"f": audio_format}

    if int(num_streams) > 1:  # video has audio stream
        try:
            video = ffmpeg.input(video)
            (ffmpeg.output(video.audio, path, **ffmpeg_args).run(capture_stderr=True))
        except ffmpeg.Error as _:
            path = None
    else:
        path = None
    return path


def sub_to_dict(sub, dedupe=True, single=False) -> list:
    """Convert WebVTT to JSON, optionally removing duplicate lines"""

    captions = webvtt.read_buffer(io.StringIO(sub))
    dicts = [{"start": c.start, "end": c.end, "lines": c.lines} for c in captions]
    if dedupe:
        dicts = []
        prev_line = None
        for c in captions:
            if any("<c>" in l for l in c.lines):
                continue
            # Collect lines that are not dupes
            not_dupe_lines = []
            for line in c.lines:
                if not line.strip():
                    continue
                if line != prev_line:
                    not_dupe_lines.append(line)
                prev_line = line
            if not_dupe_lines:
                dicts.append({"start": c.start, "end": c.end, "lines": not_dupe_lines})
    if single:
        for d in dicts:
            d["line"] = "\n".join(d.pop("lines"))
    return dicts


def get_yt_meta(url, yt_metadata_args: dict) -> dict:
    """Return yt meta dict with meta data and/or subtitles
    yt_metadata_args is a dict of follwing format:
    yt_metadata_args = {
        'writesubtitles': True,
        'subtitleslangs': ['en'],
        'writeautomaticsub': True,
        'get_info': True
    }

    writesubtitles:    Whether to write subtitles
    writeautomaticsub: Write the automatically generated subtitles to a file
    subtitleslangs:    List of languages of the subtitles to download.
    get_info: whether to add info (title, description, tags etc) to the output.

    """

    write_subs = yt_metadata_args.get("writesubtitles", None)

    yt_metadata_args["skip_download"] = True
    yt_metadata_args["ignoreerrors"] = True
    yt_metadata_args["quiet"] = True

    info_dict, sub_dict = None, None

    with yt_dlp.YoutubeDL(yt_metadata_args) as yt:

        info_dict = yt.extract_info(url, download=False)
        if write_subs:
            sub_url = info_dict["requested_subtitles"][yt_metadata_args["subtitleslangs"][0]]["url"]
            res = requests.get(sub_url)
            sub = io.TextIOWrapper(io.BytesIO(res.content)).read()
            sub_dict = sub_to_dict(sub)

        if yt_metadata_args["get_info"]:
            info_dict.pop("subtitles")
            info_dict.pop("requested_formats")
            info_dict.pop("formats")
            info_dict.pop("thumbnails")
            info_dict.pop("automatic_captions")
        else:
            info_dict = None

        yt_meta_dict = {"info": info_dict, "subtitles": sub_dict}

        return yt_meta_dict


def get_web_file_info(url):
    """returns info about the url (currently extension and modality)"""
    # TODO: make this nicer
    video_extensions = ["mp4", "webm", "mov", "avi", "mkv"]
    audio_extensions = ["mp3", "wav", "m4a"]
    for ext in video_extensions:
        if url.endswith(f".{ext}"):
            return ext, "video"
    for ext in audio_extensions:
        if url.endswith(f".{ext}"):
            return ext, "audio"
    return None


class WebFileDownloader:
    """Downloader class for mp4 links"""

    def __init__(self, timeout, tmp_dir, encode_formats):
        self.timeout = timeout
        self.tmp_dir = tmp_dir
        self.encode_formats = encode_formats

    def __call__(self, url):
        modality_paths = {}
        resp = requests.get(url, stream=True, timeout=self.timeout)
        ext, modality = get_web_file_info(url)
        modality_path = f"{self.tmp_dir}/{str(uuid.uuid4())}.{ext}"
        with open(modality_path, "wb") as f:
            f.write(resp.content)
        modality_paths[modality] = modality_path

        if modality == "video" and self.encode_formats.get("audio", None):
            audio_format = self.encode_formats["audio"]
            audio_path = video2audio(modality_paths["video"], audio_format, self.tmp_dir)
            if audio_path is not None:
                modality_paths["audio"] = audio_path

        for modality, modality_path in modality_paths.items():
            if modality not in self.encode_formats:
                os.remove(modality_path)
                modality_path.pop(modality)

        return modality_paths, None


class YtDlpDownloader:
    """Downloader class for yt-dlp links"""

    # TODO: maybe we just include height and width in the metadata_args
    def __init__(self, tmp_dir, metadata_args, video_size, audio_rate, encode_formats):
        self.tmp_dir = tmp_dir
        self.metadata_args = metadata_args
        self.video_size = video_size
        self.audio_rate = audio_rate
        self.encode_formats = encode_formats

    def __call__(self, url):
        modality_paths = {}

        # video_format_string = f"bv*[height<={self.video_size}][ext=mp4]/b[height<={self.video_size}][ext=mp4] / wv/w[ext=mp4]"
        video_format_string = (
            f"wv*[height>={self.video_size}][ext=mp4]/w[height>={self.video_size}][ext=mp4] / bv/b[ext=mp4]"
        )
        audio_fmt_string = (
            f"wa[asr>={self.audio_rate}][ext=m4a] / ba[ext=m4a]" if self.audio_rate > 0 else "ba[ext=m4a]"
        )

        if self.encode_formats.get("audio", None):
            audio_path_m4a = f"{self.tmp_dir}/{str(uuid.uuid4())}.m4a"
            ydl_opts = {
                "outtmpl": audio_path_m4a,
                "format": audio_fmt_string,
                "quiet": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download(url)

            # TODO: look into this, don't think we can just do this
            # TODO: just figure out a way to download the preferred extension using yt-dlp
            # audio_path = audio_path_m4a.replace(".m4a", f".{self.encode_formats['audio']}")
            audio_path = audio_path_m4a
            modality_paths["audio"] = audio_path

        if self.encode_formats.get("video", None):
            video_path = f"{self.tmp_dir}/{str(uuid.uuid4())}.mp4"
            modality_paths["video"] = video_path
            ydl_opts = {
                "outtmpl": video_path,
                "format": video_format_string,
                "quiet": True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download(url)
        
        if self.encode_formats.get("sb", None):
            video_path = []
            video = YouTubeFrames(url)
            sb_dict = {}
            for i, frames_image in enumerate(video.get_thumbframes()):
                file_content = frames_image.get_image()
                sb_dict[i] = [frames_image.width, frames_image.height, frames_image.cols, frames_image.rows, frames_image.n_frames, frames_image.mime_type]
                with open(f"{self.tmp_dir}/{video.video_id}_{i}.{frames_image.mime_type}", "wb") as f:
                    f.write(file_content)
                    video_path.append(f"{self.tmp_dir}/{video.video_id}_{i}.{frames_image.mime_type}")
            modality_paths["sb"] = video_path


        if self.metadata_args:
            yt_meta_dict = get_yt_meta(url, self.metadata_args)
        else:
            yt_meta_dict = {}

        yt_meta_dict["sb"] = sb_dict

        return modality_paths, yt_meta_dict, None


class VideoDataReader:
    """Video data reader provide data for a video"""

    def __init__(self, video_size, audio_rate, dl_timeout, tmp_dir, yt_meta_args, encode_formats) -> None:
        self.webfile_downloader = WebFileDownloader(dl_timeout, tmp_dir, encode_formats)
        self.yt_downloader = YtDlpDownloader(tmp_dir, yt_meta_args, video_size, audio_rate, encode_formats)

    def __call__(self, row):
        key, url = row

        meta_dict = None
        # TODO: make nice function to detect what type of link we're dealing with
        if get_web_file_info(url):  # web file that can be directly downloaded
            modality_paths, error_message = self.webfile_downloader(url)
        elif "youtube" in url:  # youtube link
            try:
                modality_paths, meta_dict, error_message = self.yt_downloader(url)
            except Exception as e:  # pylint: disable=(broad-except)
                modality_paths, meta_dict, error_message = {}, None, str(e)
        else:
            modality_paths, error_message = {}, "Warning: Unsupported URL type"

        streams = {}
        for modality, modality_path in modality_paths.items():
            if modality == "sb":
                streams[modality] = []
                for p in modality_path:
                    with open(p, "rb") as modality_file:
                        streams[modality].append(modality_file.read())
                    try:
                        os.remove(p)
                    except:
                        print("remove error")
            else:
                with open(modality_path, "rb") as modality_file:
                    streams[modality] = modality_file.read()
                os.remove(modality_path)
            
        return key, streams, meta_dict, error_message
