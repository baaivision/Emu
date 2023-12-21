### Download YouTube metadata & subtitles:
#### Usage

This is how you can download a list of youtube videos along with the associated youtube metadata using video2dataset. You must pass in a yt_metadata_args parameter which will be used to specify what information you want in your json metadata in the output dataset. If you set the 'captions_are_subtitles' parameter (as is done in the example) video2dataset will take the video (and/or audio) and split it up according to the subtitles provided by youtube and make the text caption (txt file) of that given sample clip be the subtitle of the clip.

```py
if __name__ == '__main__':

    yt_metadata_args = {
        'writesubtitles': True, # whether to write subtitles to a file
        'subtitleslangs': ['en'], # languages of subtitles (right now support only one language)
        'writeautomaticsub': True, # whether to write automatic subtitles
        'get_info': True # whether to save a video meta data into the output JSON file
    }

    video2dataset(
        url_list='input.parquet',
        input_format='parquet',
        output_format='files',
        output_folder='audio',
        yt_metadata_args=yt_metadata_args,
	captions_are_subtitles=True,
        encode_formats={"video": "mp4", "audio": "mp3"},
    )
```

#### Output

For every sample the metadata will be present in the json file as such:

```json
{
    "description": "For the past five years, King Fish has been creating a media channel for IBM to generate leads of senior IT decision makers and retain current customers.  We produce dozens of webcasts every year for numerous divisions within IBM. King Fish provides managed services, original content and audience development. \n\nKFM worked with IBM to develop video content on how SPSS Statistics can help their clients meet business goals with advanced data insight methods. The result? Much more effective than an info-graphic.",
    "videoID": "QW3-5OuWn4M",
    "start": 56.1025,
    "end": 66.10249999999999,
    "caption": "IBM SPSS",
    "url": "http://youtube.com/watch?v=QW3-5OuWn4M",
    "key": "000000_00001",
    "status": "success",
    "error_message": null,
    "yt_meta_dict": {
        "info": {
            "id": "QW3-5OuWn4M",
            "title": "IBM SPSS",
            "thumbnail": "https://i.ytimg.com/vi/QW3-5OuWn4M/maxresdefault.jpg",
            "description": "For the past five years, King Fish has been creating a media channel for IBM to generate leads of senior IT decision makers and retain current customers.  We produce dozens of webcasts every year for numerous divisions within IBM. King Fish provides managed services, original content and audience development. \n\nKFM worked with IBM to develop video content on how SPSS Statistics can help their clients meet business goals with advanced data insight methods. The result? Much more effective than an info-graphic.",
            "uploader": "King Fish Media",
            "uploader_id": "KingFishMediaBoston",
            "uploader_url": "http://www.youtube.com/user/KingFishMediaBoston",
            "channel_id": "UCDy7Xb5vYxbmSosQmztCCcQ",
            "channel_url": "https://www.youtube.com/channel/UCDy7Xb5vYxbmSosQmztCCcQ",
            "duration": 122,
            "view_count": 116,
            "average_rating": null,
            "age_limit": 0,
            "webpage_url": "https://www.youtube.com/watch?v=QW3-5OuWn4M",
            "categories": [
                "Science & Technology"
            ],
            "tags": [
                "IBM",
                "technology",
                "statistics",
                "data",
                "analysis",
                "computers",
                "content marketing",
                "Software"
            ],
            "playable_in_embed": true,
            "live_status": "not_live",
            "release_timestamp": null,
            "comment_count": null,
            "chapters": null,
            "like_count": 1,
            "channel": "King Fish Media",
            "channel_follower_count": 10,
            "upload_date": "20131107",
            "availability": "public",
            "original_url": "http://youtube.com/watch?v=QW3-5OuWn4M",
            "webpage_url_basename": "watch",
            "webpage_url_domain": "youtube.com",
            "extractor": "youtube",
            "extractor_key": "Youtube",
            "playlist": null,
            "playlist_index": null,
            "display_id": "QW3-5OuWn4M",
            "fulltitle": "IBM SPSS",
            "duration_string": "2:02",
            "is_live": false,
            "was_live": false,
            "requested_subtitles": {
                "en": {
                    "ext": "vtt",
                    "url": "https://www.youtube.com/api/timedtext?v=QW3-5OuWn4M&caps=asr&xoaf=5&hl=en&ip=0.0.0.0&ipbits=0&expire=1676200746&sparams=ip%2Cipbits%2Cexpire%2Cv%2Ccaps%2Cxoaf&signature=A43F4C223A9DBC7E3BFBC61027FC5AF70D709AB5.B386EB52DD412DEFC3E8DBBCF7F30C442473CDA4&key=yt8&kind=asr&lang=en&fmt=vtt",
                    "name": "English"
                }
            },
            "_has_drm": null,
            "format": "137 - 1920x1080 (1080p)+251 - audio only (medium)",
            "format_id": "137+251",
            "ext": "mkv",
            "protocol": "https+https",
            "language": null,
            "format_note": "1080p+medium",
            "filesize_approx": 12831366,
            "tbr": 841.009,
            "width": 1920,
            "height": 1080,
            "resolution": "1920x1080",
            "fps": 30,
            "dynamic_range": "SDR",
            "vcodec": "avc1.640028",
            "vbr": 691.069,
            "stretched_ratio": null,
            "acodec": "opus",
            "abr": 149.94,
            "asr": 48000,
            "audio_channels": 2
        }
    },
    "clips": [
        [
            "00:00:07.749",
            "00:00:07.759"
        ]
    ] 
}
```

And since we specified that captions_are_subtitles the txt file will have the subtitle for that given clip inside of it. For this particular example it would be: "analytics to assess performance based on"
