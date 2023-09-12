from youtube_transcript_api import YouTubeTranscriptApi
import os
from youtube_transcript_api.formatters import JSONFormatter

from tqdm import tqdm
from multiprocessing import Process

def save_subtitle(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
    formatter = JSONFormatter()

    json_formatted = formatter.format_transcript(transcript)

    with open(f'subtitle/{video_id}.json', 'w', encoding='utf-8') as json_file:
        json_file.write(json_formatted)


def download(video_ids, i):
    for v in tqdm(video_ids):
        try:
            save_subtitle(v)
        except:
            print("error:", v)


if __name__ == '__main__':
    num_workers = 8
    video_ids = []


    # yt urls
    with open("../YT-1B-url.txt", 'r') as f:
        urls = f.readlines()
        video_ids = [u.replace("https://www.youtube.com/watch?v=", "").strip() for u in urls]

    print("video_ids", len(video_ids), video_ids[-1], video_ids[-2])
    
    video_ids = video_ids

    processes = [Process(target=download, args=(video_ids[i::num_workers],i)) for i in range(num_workers)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


# save_subtitle("Lj4jS6qFVPo")
