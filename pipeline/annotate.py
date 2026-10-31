import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import re
import cv2
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from ultralytics import YOLO
import yt_dlp

VIDEOS_SCHEMA = pa.schema(
    [
        ("video_id", pa.string()),
        ("title", pa.string()),
        ("status", pa.string()),
        ("error", pa.string()),
        ("conf", pa.float32()),
        ("duration_sec", pa.float32()),
        ("video_fps", pa.float32()),
        ("processed_fps", pa.float32()),
        ("total_frames", pa.int64()),
        ("processed_frames", pa.int64()),
        ("width", pa.int32()),
        ("height", pa.int32()),
        ("started_at", pa.timestamp("us")),
        ("finished_at", pa.timestamp("us")),
    ]
)

DETECTIONS_SCHEMA = pa.schema(
    [
        ("video_id", pa.string()),
        ("frame_idx", pa.int64()),
        ("timestamp_sec", pa.float32()),
        ("brand_id", pa.int16()),
        ("brand_name", pa.string()),
        ("confidence", pa.float32()),
        ("cx", pa.float32()),
        ("cy", pa.float32()),
        ("w", pa.float32()),
        ("h", pa.float32()),
        ("screen_pct", pa.float32()),
        ("angle_deg", pa.float32()),
    ]
)


def save_video_metadata(output_dir: Path, meta: dict) -> Path | None:
    out = output_dir / "videos.parquet"
    now_utc = datetime.now(timezone.utc)
    row = {
        "video_id": str(meta.get("video_id", "")),
        "title": str(meta.get("title", "")),
        "status": str(meta.get("status", "completed")),
        "error": str(meta.get("error")) if meta.get("error") is not None else None,
        "conf": float(meta.get("conf", 0.25) if meta.get("conf") is not None else 0.25),
        "duration_sec": float(meta.get("duration_sec", 0.0) or 0.0),
        "video_fps": float(meta.get("video_fps", 30.0) or 30.0),
        "processed_fps": float(meta.get("processed_fps", 30.0) or 30.0),
        "total_frames": int(meta.get("total_frames", 0) or 0),
        "processed_frames": int(meta.get("processed_frames", 0) or 0),
        "width": int(meta.get("width", 0) or 0),
        "height": int(meta.get("height", 0) or 0),
        "started_at": meta.get("started_at") or now_utc,
        "finished_at": meta.get("finished_at") or now_utc,
    }
    new_tbl = pa.Table.from_pylist([row], schema=VIDEOS_SCHEMA)
    if out.exists():
        try:
            existing = []
            for r in pq.read_table(out).to_pylist():
                if r.get("video_id") != row["video_id"]:
                    r.setdefault("status", "completed")
                    r.setdefault("error", None)
                    r.setdefault("conf", row["conf"])
                    existing.append(r)
            new_tbl = pa.Table.from_pylist(existing + [row], schema=VIDEOS_SCHEMA)
        except Exception as e:
            print(f"Error reading existing metadata file {out}: {e}. Skipping write to avoid data loss.")
            return None
    pq.write_table(new_tbl, out, compression="zstd")
    return out


def save_detections(output_dir: Path, video_id: str, rows: list[dict]) -> Path:
    detections_dir = output_dir / "detections"
    detections_dir.mkdir(parents=True, exist_ok=True)
    out = detections_dir / f"{video_id}.parquet"
    table = pa.Table.from_pylist(rows, schema=DETECTIONS_SCHEMA)
    pq.write_table(table, out, compression="zstd")
    return out


def normalize_youtube_url(source: str) -> str:
    s = source.strip()
    if s.startswith(("www.", "youtube.com", "youtu.be", "m.youtube.com")):
        s = "https://" + s
    return s


def is_youtube_url(source: str) -> bool:
    """Checks whether the source string is a YouTube video or playlist URL."""
    s = normalize_youtube_url(source).lower()
    return s.startswith(("http://", "https://")) and (
        "youtube.com" in s or "youtu.be" in s
    )


def extract_yt_id(url: str) -> str:
    """Best-effort extraction of video ID from a YouTube URL."""
    m = re.search(r"(?:v=|\/embed\/|\/v\/|youtu\.be\/|\/shorts\/)([a-zA-Z0-9_-]{11})", url)
    if m:
        return m.group(1)
    return "video"


def get_video_items(source: str) -> list[str]:
    """Resolves a YouTube URL into a list of individual video URLs (single video or playlist)."""
    source = normalize_youtube_url(source)
    if not is_youtube_url(source):
        raise ValueError(
            f"Invalid source: '{source}'. Only YouTube video and playlist URLs are supported (local files/directories are not accepted)."
        )

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(source, download=False)
        if "entries" in info:
            urls = []
            for entry in info["entries"]:
                if entry:
                    url = entry.get("url")
                    if not url and entry.get("id"):
                        url = f"https://www.youtube.com/watch?v={entry.get('id')}"
                    if url:
                        urls.append(url)
            return urls
        return [source]


def resolve_video_stream(source: str):
    """Extracts direct stream URL from YouTube."""
    source = normalize_youtube_url(source)
    if not is_youtube_url(source):
        raise ValueError(
            f"Invalid source: '{source}'. Only YouTube video and playlist URLs are supported."
        )

    ydl_opts = {
        "format": "bestvideo[ext=mp4]/bestvideo/best",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(source, download=False)
        stream_url = info.get("url")
        if not stream_url:
            raise RuntimeError(f"Failed to extract YouTube stream URL for: {source}")
        return (
            stream_url,
            str(info.get("id") or "video"),
            str(info.get("title") or "video"),
            info.get("duration"),
        )


def process_video(
    model,
    output_dir: Path,
    source: str,
    target_fps: float = None,
    conf: float = 0.25,
):
    """Processes, detects brands, renders annotations, and saves Parquet for a single video."""
    started_at = datetime.now(timezone.utc)
    safe_id = extract_yt_id(source)
    title = source
    duration = None
    video_fps = 0.0
    processed_fps = 0.0
    eff_stride = 1
    w, h = 0, 0
    frame_idx = 0
    processed_frames_count = 0
    cap = None
    writer = None

    try:
        stream_url, vid_id, title, duration = resolve_video_stream(source)
        safe_id = "".join(c for c in vid_id if c.isalnum() or c in "-_") or safe_id

        device = "0" if torch.cuda.is_available() else "cpu"
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            err_msg = f"Could not open video stream for source: {source}"
            print(f"Warning: {err_msg}")
            save_video_metadata(
                output_dir,
                {
                    "video_id": safe_id,
                    "title": title,
                    "status": "failed",
                    "error": err_msg,
                    "conf": conf,
                    "duration_sec": float(duration or 0.0),
                    "video_fps": 0.0,
                    "processed_fps": 0.0,
                    "total_frames": 0,
                    "processed_frames": 0,
                    "width": 0,
                    "height": 0,
                    "started_at": started_at,
                    "finished_at": datetime.now(timezone.utc),
                },
            )
            return

        w, h = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (
            int(duration * video_fps) if duration else None
        )

        # Determine frame stride based on target_fps
        if target_fps is not None and target_fps > 0:
            eff_stride = max(1, round(video_fps / target_fps))
        else:
            eff_stride = 1

        processed_fps = max(0.1, float(video_fps / eff_stride))
        out_video = output_dir / f"{safe_id}.mp4"
        writer = cv2.VideoWriter(
            str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), processed_fps, (w, h)
        )

        detections = []
        frame_idx = 0
        processed_frames_count = 0

        print(f"\nProcessing: '{title}'")
        print(
            f"[{w}x{h} @ video_fps={video_fps:.1f} -> processed_fps={processed_fps:.1f} (stride={eff_stride}) | Device: {device}]"
        )

        with (
            torch.inference_mode(),
            tqdm(total=total, desc="Annotating", unit="frames") as pbar,
        ):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                ts = round(frame_idx / video_fps, 3)

                last_res = model.predict(
                    frame,
                    conf=conf,
                    device=device,
                    verbose=False,
                )[0]
                if last_res.obb is not None and len(last_res.obb):
                    classes = last_res.obb.cls.cpu().tolist()
                    confs = last_res.obb.conf.cpu().tolist()
                    xywhr = last_res.obb.xywhr.cpu().tolist()
                    for cls_id, c, (cx, cy, bw, bh, r) in zip(classes, confs, xywhr):
                        brand_name = model.names.get(int(cls_id), str(cls_id))
                        detections.append(
                            {
                                "video_id": safe_id,
                                "frame_idx": frame_idx,
                                "timestamp_sec": ts,
                                "brand_id": int(cls_id),
                                "brand_name": str(brand_name),
                                "confidence": float(c),
                                "cx": float(cx),
                                "cy": float(cy),
                                "w": float(bw),
                                "h": float(bh),
                                "screen_pct": float((bw * bh) / (w * h) * 100.0),
                                "angle_deg": float(math.degrees(r)),
                            }
                        )
                annotated_frame = last_res.plot(
                    line_width=1, font_size=0.4, labels=True, conf=True
                )
                writer.write(annotated_frame)
                frame_idx += 1
                processed_frames_count += 1
                pbar.update(1)

                # Skip intermediate non-sampled frames without full decoding
                for _ in range(eff_stride - 1):
                    if not cap.grab():
                        break
                    frame_idx += 1
                    pbar.update(1)

        cap.release()
        cap = None
        writer.release()
        writer = None
        print(f"Video saved: {out_video}")

        save_video_metadata(
            output_dir,
            {
                "video_id": safe_id,
                "title": title,
                "status": "completed",
                "error": None,
                "conf": conf,
                "duration_sec": float(frame_idx / video_fps) if video_fps > 0 else 0.0,
                "video_fps": float(video_fps),
                "processed_fps": float(processed_fps),
                "total_frames": int(frame_idx),
                "processed_frames": int(processed_frames_count),
                "width": int(w),
                "height": int(h),
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc),
            },
        )
        parquet_path = save_detections(output_dir, safe_id, detections)
        print(f"Saved {len(detections)} detections: {parquet_path}")

    except Exception as e:
        print(f"Error processing video '{source}': {e}")
        save_video_metadata(
            output_dir,
            {
                "video_id": safe_id,
                "title": title,
                "status": "failed",
                "error": str(e),
                "conf": conf,
                "duration_sec": float(frame_idx / video_fps) if video_fps > 0 else float(duration or 0.0),
                "video_fps": float(video_fps),
                "processed_fps": float(processed_fps),
                "total_frames": int(frame_idx),
                "processed_frames": int(processed_frames_count),
                "width": int(w),
                "height": int(h),
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc),
            },
        )
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
        if writer is not None:
            writer.release()


def run_pipeline(
    source: str,
    fps: float = None,
    conf: float = 0.25,
):
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / "ExposureEngine.pt"
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))

    video_items = get_video_items(source)
    total_videos = len(video_items)
    print(f"Found {total_videos} video(s) to process.")

    for i, item in enumerate(video_items, 1):
        if total_videos > 1:
            print(f"\n--- Video [{i}/{total_videos}] ---")
        process_video(
            model,
            output_dir,
            item,
            target_fps=fps,
            conf=conf,
        )


def main():
    parser = argparse.ArgumentParser(prog="annotate.py")
    parser.add_argument(
        "--source",
        required=True,
        metavar="URL",
        help="YouTube URL (video or playlist)",
    )
    parser.add_argument(
        "--fps", type=float, metavar="N", help="frames/sec to process (default: native)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        metavar="0-1",
        help="min detection confidence (default: %(default)s)",
    )
    args = parser.parse_args()

    run_pipeline(
        args.source,
        fps=args.fps,
        conf=args.conf,
    )


if __name__ == "__main__":
    main()
