import hashlib
import math
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import duckdb
import pandas as pd
import torch

from tqdm import tqdm
from ultralytics import YOLO

MODEL_PATH = "models/ExposureEngineEnhanced.pt"
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
DB_PATH = OUTPUT_DIR / "beams.duckdb"
CONF = 0.5
IMGSZ = 1280
JPEG_QUALITY = 90
SKIP_DONE = True
WRITER_THREADS = 4

DET_COLS = [
    "video",
    "frame",
    "brand",
    "confidence",
    "center_x",
    "center_y",
    "box_width",
    "box_height",
    "angle_deg",
]


def init_db(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            video     VARCHAR,
            file_name VARCHAR,
            fps       DOUBLE,
            duration  DOUBLE,
            width     INTEGER,
            height    INTEGER
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            video      VARCHAR,
            frame      VARCHAR,
            brand      VARCHAR,
            confidence DOUBLE,
            center_x   DOUBLE,
            center_y   DOUBLE,
            box_width  DOUBLE,
            box_height DOUBLE,
            angle_deg  DOUBLE
        )
    """)


def make_id(video):
    return hashlib.sha1(video.name.encode("utf-8")).hexdigest()[:10]


def video_info(video):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (fps if fps and fps > 0 else 30.0), total, width, height


def write_frame(r, path):
    cv2.imwrite(str(path), r.plot(), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def save_video(con, vid, file_name, fps, duration, width, height, rows):
    con.execute("DELETE FROM detections WHERE video = ?", [vid])
    con.execute("DELETE FROM videos WHERE video = ?", [vid])
    con.execute(
        "INSERT INTO videos VALUES (?, ?, ?, ?, ?, ?)",
        [vid, file_name, fps, duration, width, height],
    )
    if rows:
        df = pd.DataFrame(rows, columns=DET_COLS)
        con.execute("INSERT INTO detections SELECT * FROM df")


def process_video(model, con, pool, video):
    vid = make_id(video)
    if (
        SKIP_DONE
        and con.execute("SELECT 1 FROM videos WHERE video = ?", [vid]).fetchone()
    ):
        return 0

    out_dir = OUTPUT_DIR / vid
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    shutil.copy2(video, out_dir / f"{vid}{video.suffix.lower()}")

    fps, est_total, width, height = video_info(video)

    results = model.predict(
        source=str(video),
        stream=True,
        conf=CONF,
        imgsz=IMGSZ,
        half=torch.cuda.is_available(),
        verbose=False,
    )

    rows = []
    pending = deque()
    total_frames = 0
    saved = 0
    bar = tqdm(total=est_total or None, desc=video.name[:25], unit="frame", leave=False)

    for frame_no, r in enumerate(results):
        total_frames += 1
        bar.update(1)
        if r.obb is None or len(r.obb) == 0:
            continue

        name = f"frame_{frame_no:08d}.jpg"
        pending.append(pool.submit(write_frame, r, out_dir / name))
        if len(pending) > 32:
            pending.popleft().result()
        saved += 1

        for cls_id, conf, (cx, cy, w, h, ang) in zip(
            r.obb.cls.tolist(), r.obb.conf.tolist(), r.obb.xywhr.tolist()
        ):
            rows.append(
                (
                    vid,
                    name,
                    model.names[int(cls_id)],
                    round(conf, 4),
                    round(cx, 1),
                    round(cy, 1),
                    round(w, 1),
                    round(h, 1),
                    round(math.degrees(ang), 1),
                )
            )

    bar.close()
    for f in pending:
        f.result()

    con.begin()
    try:
        save_video(con, vid, video.name, fps, total_frames / fps, width, height, rows)
        con.commit()
    except Exception:
        con.rollback()
        raise
    return saved


def main():
    exts = {
        ".mp4",
        ".mkv",
        ".webm",
        ".avi",
        ".mov",
        ".flv",
        ".wmv",
        ".m4v",
        ".mpg",
        ".mpeg",
        ".ts",
    }
    if not INPUT_DIR.is_dir():
        print("Input folder not found:", INPUT_DIR)
        return
    videos = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in exts)
    if not videos:
        print("No video files found in", INPUT_DIR)
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    model = YOLO(MODEL_PATH)
    con = duckdb.connect(str(DB_PATH))
    init_db(con)

    total = 0
    failed = []
    with ThreadPoolExecutor(max_workers=WRITER_THREADS) as pool:
        bar = tqdm(videos, desc="Videos", unit="video")
        for video in bar:
            bar.set_postfix(file=video.name[:25], frames=total)
            try:
                total += process_video(model, con, pool, video)
            except Exception as e:
                failed.append(video.name)
                tqdm.write(f"Failed {video.name}: {e}")

    con.close()
    print("Done. Total frames saved:", total)
    if failed:
        print("Failed videos:", ", ".join(failed))


if __name__ == "__main__":
    main()
