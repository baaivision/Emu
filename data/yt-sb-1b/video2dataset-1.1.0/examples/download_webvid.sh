#!/bin/bash

wget -nc http://www.robots.ox.ac.uk/~maxbain/webvid/results_2M_val.csv

video2dataset --url_list="results_2M_val.csv" \
        --input_format="csv" \
        --output-format="webdataset" \
	--output_folder="dataset" \
        --url_col="contentUrl" \
        --caption_col="name" \
        --save_additional_columns='[videoid,page_idx,page_dir,duration]' \
        --enable_wandb=True \
        --video_size=360 \
        --number_sample_per_shard=1000 \
        --processes_count 10 
