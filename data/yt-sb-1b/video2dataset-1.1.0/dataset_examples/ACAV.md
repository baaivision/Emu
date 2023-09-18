# [ACAV100M](https://acav100m.github.io)
ACAV100M is a dataset of 100M videos with high audio-visual correspondance.

## Download the metdata
Go to the [downloads](https://acav100m.github.io/#downloads) section of the dataset page and download the subset you wish. This just has the youtube ID, start time, and end time. For the current version of video2dataset you need to preprocess this file a bit:
1. Create a column with the link instead of just the ID: f"youtube.com/watch?v={youtubeID}"
```python
df["video_link"] = df.apply(lambda row: f"https://youtube.com/watch?v={row['videoID']}", axis=1)
```
2. Create a column for the timespan consistent with video2dataset expected clip format: [[start, end]]
```python
df["clips"] = df.apply(lambda row: [[row['start'], row['end']]], axis=1)
```

## Download the videos with video2dataset

This command runs video2dataset on the input table and saves the video clips along with the metadata in the webdataset format.

```python

# args specify we want to get all metadata from the video and save to the json component of the sample
# this includes the title and description of the video
yt_metadata_args = {
	"get_info": True,  # whether to save a video meta data into the output JSON file
}

video2dataset(
	url_list="/admin/home-iejmac/datasets/acav100m/ACAV100M_clip_unique.parquet",
	output_folder="s3://s-laion/acav100m/mp4_acav100m",
	output_format="webdataset",
	input_format="parquet",
	url_col="video_link",
	caption_col="title",
	clip_col="clips",
	save_additional_columns=["videoID", "start", "end"],
	enable_wandb=True,
	video_size=360,
	number_sample_per_shard=2000,
	subjob_size=10000,
	processes_count=96,
	thread_count=48,
	distributor="pyspark",
	yt_metadata_args=yt_metadata_args
)
```

## Performance
The job ran on 64 nodes with 48 CPU cores each and processed the dataset in 7 hours. This wasn't the entire ACAV100M dataset but instead a 60M subset of only the unique videos in the dataset. The title and description of the videos were pre-extracted and placed in the parquet file along with the description and also the timespan of the clip was processed into the format that video2dataset expects. [Here's](https://wandb.ai/iejmac/video2dataset/runs/3kon2409?workspace=user-iejmac) a link to the wandb logs from the processing of the dataset. We were able to get 0.8 vid/s/core which comes out to approximately 100MB/s/core.
