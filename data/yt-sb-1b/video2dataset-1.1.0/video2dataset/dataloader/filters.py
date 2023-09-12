"""WebDataset filters"""
from langdetect import detect_langs, DetectorFactory  # pylint: disable=unused-import

from webdataset.autodecode import decoders  # pylint: disable=unused-import


class LanguageFilter:
    """Filters the dataset based on the language"""

    def __init__(self, languages="en", lang_key="txt"):
        self.languages = languages
        if not isinstance(self.languages, list) and self.languages is not None:
            self.languages = [self.languages]

        self.lang_key = lang_key

    def __call__(self, x):
        valid = True
        if self.languages:
            try:
                valid = False
                for k in self.languages:
                    # langs = detect_langs(decoders[k](x[k]))
                    langs = detect_langs(x[k])
                    valid |= max(langs, key=lambda x: x.prob).lang in self.languages
            except Exception:  # pylint: disable=broad-except
                valid = False
        return valid


class KeyFilter:
    """Filters the dataset based on the key"""

    def __init__(self, video_key="mp4"):
        self.video_key = video_key

    def __call__(self, sample):
        try:
            return self.video_key in sample and "txt" in sample
        except Exception as _:  # pylint: disable=broad-except
            return False


class AestheticsFilter:
    """Filters the dataset based on aesthethics"""

    def __init__(self, aesthetic_thld=None, aesthetic_key="AESTHETIC_SCORE"):
        self.aesthetic_thld = aesthetic_thld
        self.aesthetic_key = aesthetic_key

    def __call__(self, sample):
        if self.aesthetic_thld is not None:
            try:
                return sample["json"][self.aesthetic_key] >= self.aesthetic_thld
            except Exception as e:  # pylint: disable=broad-except
                if self.aesthetic_key not in sample["json"]:
                    raise e
                return True
        else:
            return True


class UnsafeFilter:
    """Filters the dataset based on the probability a sample is unsafe"""

    def __init__(self, p_unsafe_threshold):
        self.p_unsafe_threshold = p_unsafe_threshold

    def __call__(self, sample):
        valid = True
        if self.p_unsafe_threshold is not None and "json " in sample:
            try:
                valid = sample["json"]["punsafe"] < self.p_unsafe_threshold
            except Exception:  # pylint: disable=broad-except
                if "punsafe" not in sample["json"]:
                    raise
                valid = False
        return valid
