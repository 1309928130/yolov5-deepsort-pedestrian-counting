# YOLOv5 + DeepSORT Pedestrian Counting

Pedestrian detection and counting from video using YOLOv5 and DeepSORT. Tracks people, counts total unique pedestrians, and counts crossings over configurable counting lines (with up/down direction).

This repository is a **derivative work** based on [zengwb-lx/yolov5-deepsort-pedestrian-counting](https://github.com/zengwb-lx/yolov5-deepsort-pedestrian-counting). The upstream project and its [blog post](https://blog.csdn.net/zengwubbb/article/details/113422048) explain the original YOLOv5 + DeepSORT line-crossing approach.

## License

This project inherits the **GPL-3.0** license from the upstream repository. See [LICENSE](LICENSE).

## What this fork adds

On top of the original single-line counter (`person_count.py`), this version includes tooling for station and research workflows:

| Script | Purpose |
|--------|---------|
| `person_count.py` | Multi-line counting (several vertical lines) with on-screen Chinese labels |
| `person_count_BacktoOrigin.py` | Single counting line with per-frame CSV export |
| `person_count_BacktoOrigin_processFolder.py` | Batch processing of a folder of videos |
| `person_count_multiLines.py` | Multiple lines with detailed per-line CSV output |
| `ENDautomation_pedestrian.py` | CSV-driven batch runs for pedestrian videos |
| `ENDautomation_car.py` | CSV-driven batch runs for vehicle videos |
| `ENDVideoSnaps.py` | Extract frames from videos |
| `ENDcountPedestrianNumbers.py` | Count pedestrians in simulation screenshots by color mask (non-YOLO) |

Other small changes vs upstream:

- NumPy compatibility fixes (`int` / `float` instead of deprecated `np.int` / `np.float`)
- Updated Pillow requirement
- Adjusted bounding-box label rendering in `utils/draw.py`
- `.gitignore` for virtualenvs, weights, and local data folders

## Setup

```bash
pip install -r requirements.txt
```

Download YOLOv5 weights (not included in this repo):

```bash
# Linux/macOS
bash weights/download_weights.sh

# Or manually place yolov5s.pt in weights/
```

DeepSORT ReID checkpoints are already under `deep_sort/deep/checkpoint/`.

## Quick start

Edit the `--video_path` default in the script you want to run, or pass it on the command line:

```bash
python person_count.py --video_path path/to/video.mp4
```

For CSV output and a single counting line:

```bash
python person_count_BacktoOrigin.py --video_path path/to/video.mp4
```

Press `q` while the preview window is focused to stop playback.

## Counting lines

Line positions are defined in code (pixel coordinates). Adjust them in the script you use—for example, vertical lines in `person_count.py` or a single gate line in `person_count_BacktoOrigin.py`.

## Automation scripts

`ENDautomation_pedestrian.py` and related `END*` scripts expect local CSV config and video paths (e.g. under `E:/videos/`). Update those paths before running on your machine.

## Acknowledgements

- [zengwb-lx/yolov5-deepsort-pedestrian-counting](https://github.com/zengwb-lx/yolov5-deepsort-pedestrian-counting) — original pedestrian counting implementation
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [Deep SORT](https://github.com/nwojke/deep_sort)
