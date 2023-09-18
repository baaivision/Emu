# YT-Storyboard-1B

Videos with interleaved subtitles text represent a valuable and scalable source of multimodal data that has received limited attention thus far. In our study, we introduced YT-Storyboard-1B dataset, which collected storyboard images from YouTube, utilizing the video-ids provided by the **YT-Temporal-1B dataset**, which encompasses a vast collection of 18 million videos, equating to a total of 1.8 billion storyboard images. Specifically, for each video, we crawl the storyboard images and subtitles files directly. Where the sampling time between storyboard images is fixed, so the start time of each image can be determined by the order. Subtitle files record the content of each subtitle, as well as the start and end times. Therefore, storyboard images and subtitles can be sorted according to their timestamps and adjacent subtitles can be merged to form an interleaved video-text sequence. By opting to collect storyboard images instead of raw video data, we eliminate the necessity of video decoding. Moreover, this approach leads to a substantial 20-fold reduction in data storage costs, resulting in increased download efficiency.




## Get the Youtube Ids

Get the Youtube Ids from YT-Temporal-1B datasets introduced by the [merlot_reserve](https://rowanzellers.com/merlotreserve/)


## Collect StoryBoard Images


We use [video2dataset](https://github.com/iejMac/video2dataset) and [thumbframes_dl](https://github.com/MarcAbonce/thumbframes_dl) to download the storyboard images of youtube video.

```shell
cd video2dataset-1.1.0
pip install -e .
video2dataset --url_list=xxx --output_folder=xxx
```


## Collect Subtitle Files

We use [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) to get the Subtitle files

```shell
python get_transcript.py
```

## Process to Video-Text interleaved Data

We process the storyboard images and subtitles according to their timestamps, and adjacent subtitles can be merged to form an interleaved video-text sequence, form as multiple image-text pairs stored in [WebDataset](https://github.com/webdataset/webdataset) format.


```shell
python video_webdataset_maker_YT1b_sb.py --input xxx --output xxx --workers 1 2>&1 | tee log.txt
```