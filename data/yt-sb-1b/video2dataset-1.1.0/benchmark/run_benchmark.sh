#!/bin/bash

rm -rf dataset

video2dataset --url_list="benchmark_vids.parquet" \
        --input_format="parquet" \
        --output_folder="dataset" \
        --output-format="files" \
        --url_col="videoLoc" \
        --caption_col="title" \
        --save_additional_columns='[videoID,description,start,end]' \
        --enable_wandb=True \
        --video_height=360 \
        --video_width=640 \
        --number_sample_per_shard=10 \
        --processes_count 10 
