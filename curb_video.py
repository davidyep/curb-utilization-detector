"""Video processing pipeline for curb utilization analysis."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Generator, Optional

import cv2
import numpy as np
import pandas as pd

from curb_config import (
    COCO_CATEGORIES,
    WORLD_CATEGORIES,
    DEFAULT_FRAME_SKIP,
    DEFAULT_BATCH_SIZE,
    DEFAULT_OUTPUT_DIR,
    OVERLAP_THRESHOLD,
)
from curb_detection import (
    StreetSceneDetector,
    Detection,
    ClassifiedDetection,
    FrameResult,
    load_roi_polygons,
    build_roi_masks,
    classify_detections,
    analyze_frame,
    draw_annotated_frame,
    _build_frame_result,
)


# ------------------------------------------------------------------ #
#  Video Frame Iterator
# ------------------------------------------------------------------ #

def get_video_metadata(video_path: str | Path) -> dict:
    """Return dict with fps, total_frames, duration_sec, width, height."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    meta = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    meta["duration_sec"] = (
        meta["total_frames"] / meta["fps"] if meta["fps"] > 0 else 0.0
    )
    cap.release()
    return meta


def iter_video_frames(
    video_path: str | Path,
    frame_skip: int = DEFAULT_FRAME_SKIP,
    max_frames: Optional[int] = None,
) -> Generator[tuple[int, float, np.ndarray], None, None]:
    """Yield (frame_index, timestamp_seconds, bgr_frame) from a video file.

    Args:
        video_path: path to .mp4, .avi, etc.
        frame_skip: yield every Nth frame (e.g., 30 = ~1 fps for 30fps video)
        max_frames: stop after this many yielded frames (None = entire video)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    yielded = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            timestamp = frame_idx / fps
            yield (frame_idx, timestamp, frame)
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break
        frame_idx += 1

    cap.release()


# ------------------------------------------------------------------ #
#  Batch Processing
# ------------------------------------------------------------------ #

def process_video(
    video_path: str | Path,
    roi_polygons: dict[str, list[tuple[int, int]]],
    detector=None,
    frame_skip: int = DEFAULT_FRAME_SKIP,
    batch_size: int = DEFAULT_BATCH_SIZE,
    overlap_threshold: float = OVERLAP_THRESHOLD,
    max_frames: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, FrameResult], None]] = None,
) -> list[FrameResult]:
    """Process an entire video file and return per-frame results.

    Args:
        video_path: path to video file
        roi_polygons: dict from load_roi_polygons()
        detector: StreetSceneDetector, CombinedDetector, etc. (created if None)
        frame_skip: process every Nth frame
        batch_size: frames per YOLO inference batch
        overlap_threshold: minimum overlap ratio
        max_frames: cap on frames to process (None = all)
        progress_callback: called after each batch with (processed, total_est, latest)
    """
    if detector is None:
        detector = StreetSceneDetector()

    meta = get_video_metadata(video_path)
    total_est = meta["total_frames"] // frame_skip
    if max_frames is not None:
        total_est = min(total_est, max_frames)

    # Build ROI masks once using the video's frame dimensions
    frame_shape = (meta["height"], meta["width"])
    roi_masks = build_roi_masks(roi_polygons, frame_shape)

    all_results: list[FrameResult] = []

    # Accumulate frames into batches
    batch_frames: list[np.ndarray] = []
    batch_meta: list[tuple[int, float]] = []  # (frame_index, timestamp)

    for frame_idx, timestamp, frame in iter_video_frames(video_path, frame_skip, max_frames):
        batch_frames.append(frame)
        batch_meta.append((frame_idx, timestamp))

        if len(batch_frames) >= batch_size:
            _process_batch(
                batch_frames, batch_meta, detector, roi_masks,
                overlap_threshold, all_results, progress_callback, total_est,
            )
            batch_frames.clear()
            batch_meta.clear()

    # Process remaining frames
    if batch_frames:
        _process_batch(
            batch_frames, batch_meta, detector, roi_masks,
            overlap_threshold, all_results, progress_callback, total_est,
        )

    return all_results


def _process_batch(
    frames: list[np.ndarray],
    meta: list[tuple[int, float]],
    detector,
    roi_masks: dict[str, np.ndarray],
    overlap_threshold: float,
    results_out: list[FrameResult],
    progress_callback: Optional[Callable],
    total_est: int,
) -> None:
    """Detect + classify a batch of frames and append to results_out."""
    batch_dets = detector.detect_batch(frames)

    for (frame_idx, timestamp), dets in zip(meta, batch_dets):
        classified = classify_detections(dets, roi_masks, overlap_threshold)
        fr = _build_frame_result(classified, frame_idx, timestamp, overlap_threshold)
        results_out.append(fr)

        if progress_callback is not None:
            progress_callback(len(results_out), total_est, fr)


# ------------------------------------------------------------------ #
#  Single Image Processing
# ------------------------------------------------------------------ #

def process_image(
    image_path: str | Path,
    roi_polygons: dict[str, list[tuple[int, int]]],
    detector=None,
    overlap_threshold: float = OVERLAP_THRESHOLD,
) -> FrameResult:
    """Process a single image file."""
    if detector is None:
        detector = StreetSceneDetector()
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to load image: {image_path}")
    roi_masks = build_roi_masks(roi_polygons, img.shape[:2])
    return analyze_frame(img, detector, roi_masks, overlap_threshold=overlap_threshold)


# ------------------------------------------------------------------ #
#  Result Aggregation
# ------------------------------------------------------------------ #

def results_to_dataframe(results: list[FrameResult]) -> pd.DataFrame:
    """Time-series DataFrame: one row per analyzed frame."""
    rows = []
    for fr in results:
        rows.append({
            "frame_index": fr.frame_index,
            "timestamp_sec": round(fr.timestamp_sec, 2),
            "total_detections": fr.total_detections,
            "total_vehicles": fr.total_vehicles,
            "pedestrians": fr.category_counts.get("pedestrian", 0),
            "cyclists": fr.category_counts.get("cyclist", 0),
            "street_infrastructure": fr.category_counts.get("street_infrastructure", 0),
            "animals": fr.category_counts.get("animal", 0),
            "personal_items": fr.category_counts.get("personal_item", 0),
            "street_furniture": fr.category_counts.get("street_furniture", 0),
            # YOLO-World categories
            "road_infrastructure": fr.category_counts.get("road_infrastructure", 0),
            "road_markings": fr.category_counts.get("road_marking", 0),
            "construction": fr.category_counts.get("construction", 0),
            "vendors": fr.category_counts.get("vendor", 0),
            "waste": fr.category_counts.get("waste", 0),
            "curb_occupied": fr.curb_occupied_count,
            "travel_occupied": fr.travel_occupied_count,
            "bike_encroachment": fr.bike_encroachment_count,
            "double_park": fr.double_park_count,
            "pedestrians_in_travel": fr.pedestrians_in_travel,
            "pedestrians_in_bike": fr.pedestrians_in_bike,
            "cyclists_in_bike": fr.cyclists_in_bike,
            "obstructions_in_bike": fr.obstructions_in_bike,
        })
    return pd.DataFrame(rows)


def detections_to_dataframe(results: list[FrameResult]) -> pd.DataFrame:
    """Detailed DataFrame: one row per detection per frame."""
    rows = []
    for fr in results:
        for cd in fr.detections:
            d = cd.detection
            row = {
                "frame_index": fr.frame_index,
                "timestamp_sec": round(fr.timestamp_sec, 2),
                "label": d.label,
                "category": d.category,
                "confidence": round(d.confidence, 3),
                "x1": d.bbox[0],
                "y1": d.bbox[1],
                "x2": d.bbox[2],
                "y2": d.bbox[3],
                "curb_occupancy": cd.curb_occupancy,
                "travel_occupancy": cd.travel_occupancy,
                "bike_encroachment": cd.bike_encroachment,
                "double_park_candidate": cd.double_park_candidate,
            }
            for zone, ratio in cd.overlap_ratios.items():
                row[f"overlap_{zone}"] = round(ratio, 3)
            rows.append(row)
    return pd.DataFrame(rows)


def compute_summary_stats(results: list[FrameResult]) -> dict:
    """Aggregate statistics across the full video."""
    if not results:
        return {}

    total_frames = len(results)
    duration = results[-1].timestamp_sec - results[0].timestamp_sec if total_frames > 1 else 0.0
    vehicle_counts = [fr.total_vehicles for fr in results]
    curb_counts = [fr.curb_occupied_count for fr in results]
    dp_frames = sum(1 for fr in results if fr.double_park_count > 0)

    # Object type breakdown across all detections
    type_counts: dict[str, int] = {}
    for fr in results:
        for cd in fr.detections:
            lbl = cd.detection.label
            type_counts[lbl] = type_counts.get(lbl, 0) + 1

    stats: dict = {
        # --- Existing keys (preserved for backward compat) ---
        "total_frames_analyzed": total_frames,
        "duration_analyzed_sec": round(duration, 1),
        "avg_vehicles_per_frame": round(sum(vehicle_counts) / total_frames, 1),
        "max_vehicles_in_frame": max(vehicle_counts),
        "total_double_park_frames": dp_frames,
        "double_park_pct": round(100.0 * dp_frames / total_frames, 1),
        "vehicle_type_breakdown": type_counts,
        "curb_utilization_pct": round(
            100.0 * sum(1 for c in curb_counts if c > 0) / total_frames, 1
        ),
        "peak_curb_occupancy": max(curb_counts),
        "avg_curb_occupancy": round(sum(curb_counts) / total_frames, 1),
    }

    # --- Category-level breakdown ---
    category_totals: dict[str, int] = {}
    for fr in results:
        for cat, cnt in fr.category_counts.items():
            category_totals[cat] = category_totals.get(cat, 0) + cnt
    stats["category_breakdown"] = category_totals

    # Per-category averages (both COCO and YOLO-World categories)
    all_category_names = list(COCO_CATEGORIES.keys()) + list(WORLD_CATEGORIES.keys())
    category_per_frame: dict[str, list[int]] = {}
    for fr in results:
        for cat in all_category_names:
            category_per_frame.setdefault(cat, []).append(fr.category_counts.get(cat, 0))
    stats["avg_per_frame_by_category"] = {
        cat: round(sum(vals) / total_frames, 1)
        for cat, vals in category_per_frame.items()
        if sum(vals) > 0
    }

    # Total detections across all categories
    all_counts = [fr.total_detections for fr in results]
    stats["avg_detections_per_frame"] = round(sum(all_counts) / total_frames, 1)
    stats["max_detections_in_frame"] = max(all_counts)

    # --- Pedestrian stats ---
    ped_counts = [fr.category_counts.get("pedestrian", 0) for fr in results]
    if sum(ped_counts) > 0:
        stats["pedestrian_stats"] = {
            "avg_per_frame": round(sum(ped_counts) / total_frames, 1),
            "max_in_frame": max(ped_counts),
            "frames_with_pedestrians_in_travel": sum(
                1 for fr in results if fr.pedestrians_in_travel > 0
            ),
            "frames_with_pedestrians_in_bike": sum(
                1 for fr in results if fr.pedestrians_in_bike > 0
            ),
        }

    # --- Bike lane stats ---
    bike_obstruction_frames = sum(1 for fr in results if fr.obstructions_in_bike > 0)
    if any(fr.cyclists_in_bike > 0 or fr.obstructions_in_bike > 0 for fr in results):
        stats["bike_lane_stats"] = {
            "avg_cyclists_in_lane": round(
                sum(fr.cyclists_in_bike for fr in results) / total_frames, 1
            ),
            "obstruction_frames": bike_obstruction_frames,
            "obstruction_pct": round(100.0 * bike_obstruction_frames / total_frames, 1),
        }

    return stats


# ------------------------------------------------------------------ #
#  Export
# ------------------------------------------------------------------ #

def export_results(
    results: list[FrameResult],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    prefix: str = "curb_analysis",
) -> dict[str, Path]:
    """Export results to CSV and JSON files.

    Creates:
        {prefix}_timeseries.csv
        {prefix}_detections.csv
        {prefix}_summary.json
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts_path = out / f"{prefix}_timeseries.csv"
    results_to_dataframe(results).to_csv(ts_path, index=False)

    det_path = out / f"{prefix}_detections.csv"
    detections_to_dataframe(results).to_csv(det_path, index=False)

    summary_path = out / f"{prefix}_summary.json"
    summary_path.write_text(
        json.dumps(compute_summary_stats(results), indent=2),
        encoding="utf-8",
    )

    return {"timeseries": ts_path, "detections": det_path, "summary": summary_path}


# ------------------------------------------------------------------ #
#  Annotated Video Export
# ------------------------------------------------------------------ #

def export_annotated_video(
    video_path: str | Path,
    results: list[FrameResult],
    roi_polygons: dict[str, list[tuple[int, int]]],
    output_path: str | Path,
    frame_skip: int = DEFAULT_FRAME_SKIP,
) -> Path:
    """Write an annotated MP4 with detection overlays for analyzed frames."""
    output_path = Path(output_path)

    # Build a lookup from frame_index -> FrameResult
    result_map = {fr.frame_index: fr for fr in results}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps_out = cap.get(cv2.CAP_PROP_FPS) / frame_skip  # output at sampled rate
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, max(1.0, fps_out), (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in result_map:
            annotated = draw_annotated_frame(frame, result_map[frame_idx], roi_polygons)
            writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()
    return output_path
