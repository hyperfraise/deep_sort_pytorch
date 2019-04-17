# Deep Sort with PyTorch

This repo allows you to easily extract all tracking results from a directory of videos.
It is a modification of this one https://github.com/ZQPei/deep_sort_pytorch. It also adds half precision inference, batch inference (though the latter doesn't improve speed on my machine).

Usage :

```
python3 track.py -h
  -h, --help            show this help message and exit                                               │
  --batch_size BATCH_SIZE, -bs BATCH_SIZE                                                             │
  -half                 FP16 inference                                                                │
  -path VIDEOS_PATH, --videos_path VIDEOS_PATH                                                        │
  -opath OUTPUT_PATH, --output_path OUTPUT_PATH                                                       │
  -ft FPS_TARGET, --fps_target FPS_TARGET                                                             │
  -wv WRITE_VIDEOS, --write_videos WRITE_VIDEOS                                                       │
  -wj WRITE_JSONS, --write_jsons WRITE_JSONS                                                          │
```

