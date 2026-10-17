from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO
import cv2

MODEL_PATH = "models/model_m.pt"
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
CONF = 0.25
FRAMES_PER_SECOND = 1
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg", ".ts"}


def get_fps(video):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else 30.0


def process_video(model, video):
    frames_dir = OUTPUT_DIR / video.stem
    frames_dir.mkdir(parents=True, exist_ok=True)

    stride = max(1, round(get_fps(video) / FRAMES_PER_SECOND))
    saved = 0

    results = model.predict(
        source=str(video),
        stream=True,
        conf=CONF,
        vid_stride=stride,
        verbose=False,
    )

    for i, r in enumerate(results):
        if r.obb is None or len(r.obb) == 0:
            continue
        cv2.imwrite(str(frames_dir / f"frame_{i * stride:06d}.jpg"), r.plot())
        saved += 1

    return saved


def main():
    videos = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        print("No video files found in", INPUT_DIR)
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    model = YOLO(MODEL_PATH)

    total = 0
    bar = tqdm(videos, desc="Videos", unit="video")
    for video in bar:
        bar.set_postfix(file=video.name[:25], frames=total)
        total += process_video(model, video)

    print("Done. Total frames saved:", total)


if __name__ == "__main__":
    main()
