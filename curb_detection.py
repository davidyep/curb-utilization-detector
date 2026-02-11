"""Core street-scene detection: YOLO object detection + ROI overlap classification."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from curb_config import (
    VEHICLE_CLASS_IDS,
    COCO_CATEGORIES,
    COCO_CLASS_LOOKUP,
    ALL_TRACKED_CLASS_IDS,
    ZONE_ANALYSIS_CATEGORIES,
    OBSTRUCTION_CATEGORIES,
    YOLO_MODEL_NAME,
    YOLO_CONFIDENCE,
    YOLO_IOU_NMS,
    OVERLAP_THRESHOLD,
    DEFAULT_ROI_PATH,
    ZONE_COLORS_RGB,
    BEHAVIOR_COLORS_RGB,
    CATEGORY_COLORS_RGB,
)


# ------------------------------------------------------------------ #
#  Data Structures
# ------------------------------------------------------------------ #

@dataclass
class Detection:
    """A single detected object."""
    label: str
    category: str  # e.g. "vehicle", "pedestrian", "street_infrastructure"
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int


@dataclass
class ClassifiedDetection:
    """A detection with zone-overlap behavior flags."""
    detection: Detection
    curb_occupancy: bool = False
    travel_occupancy: bool = False
    bike_encroachment: bool = False
    double_park_candidate: bool = False
    overlap_ratios: dict[str, float] = field(default_factory=dict)


@dataclass
class FrameResult:
    """All results for a single analyzed frame."""
    frame_index: int
    timestamp_sec: float
    detections: list[ClassifiedDetection]

    # --- Existing vehicle/zone counts (unchanged semantics) ---
    total_vehicles: int = 0
    curb_occupied_count: int = 0
    travel_occupied_count: int = 0
    bike_encroachment_count: int = 0
    double_park_count: int = 0

    # --- Total across all categories ---
    total_detections: int = 0

    # --- Per-category counts ---
    category_counts: dict[str, int] = field(default_factory=dict)

    # --- Pedestrian zone presence ---
    pedestrians_in_curb: int = 0
    pedestrians_in_travel: int = 0
    pedestrians_in_bike: int = 0

    # --- Bike lane specifics ---
    cyclists_in_bike: int = 0
    obstructions_in_bike: int = 0


# ------------------------------------------------------------------ #
#  ROI Handling
# ------------------------------------------------------------------ #

def load_roi_polygons(path: Path | str = DEFAULT_ROI_PATH) -> dict[str, list[tuple[int, int]]]:
    """Load ROI polygons from JSON (same format as roi_picker.py output).

    Returns dict mapping zone name -> list of (x, y) tuples.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ROI file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    polygons: dict[str, list[tuple[int, int]]] = {}
    for name, pts in raw.items():
        if len(pts) < 3:
            raise ValueError(f"ROI '{name}' needs at least 3 points, got {len(pts)}")
        polygons[name] = [(int(x), int(y)) for x, y in pts]
    return polygons


def build_roi_masks(
    roi_polygons: dict[str, list[tuple[int, int]]],
    frame_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Convert polygon vertices to binary masks using cv2.fillPoly.

    Args:
        roi_polygons: dict from load_roi_polygons()
        frame_shape: (height, width)

    Returns dict mapping zone name -> bool ndarray of shape (H, W).
    """
    h, w = frame_shape
    masks: dict[str, np.ndarray] = {}
    for name, pts in roi_polygons.items():
        mask = np.zeros((h, w), dtype=np.uint8)
        arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [arr], 255)
        masks[name] = mask.astype(bool)
    return masks


# ------------------------------------------------------------------ #
#  Overlap Logic
# ------------------------------------------------------------------ #

def compute_overlap(bbox: tuple[int, int, int, int], roi_mask: np.ndarray) -> float:
    """Fraction of bbox area that overlaps with roi_mask.

    Uses array slicing instead of allocating a full-frame mask per bbox.
    """
    x1, y1, x2, y2 = bbox
    h, w = roi_mask.shape
    x1c, x2c = max(0, min(w, x1)), max(0, min(w, x2))
    y1c, y2c = max(0, min(h, y1)), max(0, min(h, y2))
    bbox_area = (x2c - x1c) * (y2c - y1c)
    if bbox_area <= 0:
        return 0.0
    intersection = np.count_nonzero(roi_mask[y1c:y2c, x1c:x2c])
    return intersection / bbox_area


def classify_detections(
    detections: list[Detection],
    roi_masks: dict[str, np.ndarray],
    threshold: float = OVERLAP_THRESHOLD,
) -> list[ClassifiedDetection]:
    """Assign behavior flags to each detection based on ROI overlap.

    Zone-occupancy flags are only set for categories in ZONE_ANALYSIS_CATEGORIES.
    All detections get overlap_ratios computed regardless.
    """
    results: list[ClassifiedDetection] = []
    for det in detections:
        ratios: dict[str, float] = {}
        for zone_name, mask in roi_masks.items():
            ratios[zone_name] = compute_overlap(det.bbox, mask)

        # Only flag zone occupancy for relevant categories
        if det.category in ZONE_ANALYSIS_CATEGORIES:
            curb = ratios.get("curb", 0.0) >= threshold
            travel = ratios.get("travel", 0.0) >= threshold
            bike = ratios.get("bike", 0.0) >= threshold
        else:
            curb = travel = bike = False

        results.append(ClassifiedDetection(
            detection=det,
            curb_occupancy=curb,
            travel_occupancy=travel,
            bike_encroachment=bike,
            double_park_candidate=curb and travel and det.category == "vehicle",
            overlap_ratios=ratios,
        ))
    return results


# ------------------------------------------------------------------ #
#  YOLO Detector
# ------------------------------------------------------------------ #

class StreetSceneDetector:
    """Wraps ultralytics YOLO for NYC street-scene object detection."""

    def __init__(
        self,
        model_name: str = YOLO_MODEL_NAME,
        confidence: float = YOLO_CONFIDENCE,
        iou_threshold: float = YOLO_IOU_NMS,
        device: Optional[str] = None,
        categories: Optional[set[str]] = None,
    ):
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device

        # Build the set of tracked class IDs from selected categories
        if categories is None:
            self._tracked_ids = ALL_TRACKED_CLASS_IDS
        else:
            self._tracked_ids = {
                cls_id for cat in categories
                for cls_id in COCO_CATEGORIES.get(cat, {}).keys()
            }

    def _parse_results(self, results) -> list[list[Detection]]:
        """Convert ultralytics Results objects to lists of Detection."""
        all_dets: list[list[Detection]] = []
        for result in results:
            frame_dets: list[Detection] = []
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                all_dets.append(frame_dets)
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if cls_id not in self._tracked_ids:
                    continue
                lookup = COCO_CLASS_LOOKUP.get(cls_id)
                if lookup is None:
                    continue
                category, label = lookup
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                frame_dets.append(Detection(
                    label=label,
                    category=category,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=conf,
                    class_id=cls_id,
                ))
            all_dets.append(frame_dets)
        return all_dets

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a single BGR frame."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        parsed = self._parse_results(results)
        return parsed[0] if parsed else []

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Run inference on a batch of BGR frames."""
        results = self.model.predict(
            source=frames,
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        return self._parse_results(results)


# Backward compatibility
VehicleDetector = StreetSceneDetector


# ------------------------------------------------------------------ #
#  Frame-Level Aggregation Helper
# ------------------------------------------------------------------ #

def _build_frame_result(
    classified: list[ClassifiedDetection],
    frame_index: int,
    timestamp_sec: float,
    overlap_threshold: float = OVERLAP_THRESHOLD,
) -> FrameResult:
    """Compute all counts from a list of classified detections."""
    # Per-category counts
    category_counts: dict[str, int] = {}
    for c in classified:
        cat = c.detection.category
        category_counts[cat] = category_counts.get(cat, 0) + 1

    # Pedestrian zone presence
    ped_dets = [c for c in classified if c.detection.category == "pedestrian"]
    peds_curb = sum(1 for c in ped_dets if c.overlap_ratios.get("curb", 0) >= overlap_threshold)
    peds_travel = sum(1 for c in ped_dets if c.overlap_ratios.get("travel", 0) >= overlap_threshold)
    peds_bike = sum(1 for c in ped_dets if c.overlap_ratios.get("bike", 0) >= overlap_threshold)

    # Bike lane specifics
    cyclists_bike = sum(
        1 for c in classified
        if c.detection.category == "cyclist"
        and c.overlap_ratios.get("bike", 0) >= overlap_threshold
    )
    obstructions_bike = sum(
        1 for c in classified
        if c.detection.category in OBSTRUCTION_CATEGORIES
        and c.detection.category != "cyclist"
        and c.overlap_ratios.get("bike", 0) >= overlap_threshold
    )

    return FrameResult(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        detections=classified,
        total_vehicles=category_counts.get("vehicle", 0),
        curb_occupied_count=sum(1 for c in classified if c.curb_occupancy),
        travel_occupied_count=sum(1 for c in classified if c.travel_occupancy),
        bike_encroachment_count=sum(1 for c in classified if c.bike_encroachment),
        double_park_count=sum(1 for c in classified if c.double_park_candidate),
        total_detections=len(classified),
        category_counts=category_counts,
        pedestrians_in_curb=peds_curb,
        pedestrians_in_travel=peds_travel,
        pedestrians_in_bike=peds_bike,
        cyclists_in_bike=cyclists_bike,
        obstructions_in_bike=obstructions_bike,
    )


# ------------------------------------------------------------------ #
#  High-Level Single-Frame API
# ------------------------------------------------------------------ #

def analyze_frame(
    frame: np.ndarray,
    detector: StreetSceneDetector,
    roi_masks: dict[str, np.ndarray],
    frame_index: int = 0,
    timestamp_sec: float = 0.0,
    overlap_threshold: float = OVERLAP_THRESHOLD,
) -> FrameResult:
    """Full pipeline for one frame: detect -> classify -> aggregate."""
    dets = detector.detect(frame)
    classified = classify_detections(dets, roi_masks, overlap_threshold)
    return _build_frame_result(classified, frame_index, timestamp_sec, overlap_threshold)


# ------------------------------------------------------------------ #
#  Visualization
# ------------------------------------------------------------------ #

def _rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return (rgb[2], rgb[1], rgb[0])


def draw_annotated_frame(
    frame: np.ndarray,
    result: FrameResult,
    roi_polygons: dict[str, list[tuple[int, int]]],
    draw_zones: bool = True,
    draw_boxes: bool = True,
    draw_labels: bool = True,
) -> np.ndarray:
    """Return a copy of the frame with ROI zones, bounding boxes, and labels drawn."""
    canvas = frame.copy()

    # Draw semi-transparent zone overlays
    if draw_zones:
        overlay = canvas.copy()
        for name, pts in roi_polygons.items():
            color_rgb = ZONE_COLORS_RGB.get(name, (128, 128, 128))
            color_bgr = _rgb_to_bgr(color_rgb)
            arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [arr], color_bgr)
            cv2.polylines(canvas, [arr], isClosed=True, color=color_bgr, thickness=2)
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

    # Draw bounding boxes
    if draw_boxes:
        for cd in result.detections:
            det = cd.detection

            # Behavior-based coloring for zone-analysis objects
            if det.category in ZONE_ANALYSIS_CATEGORIES:
                if cd.double_park_candidate:
                    color_key = "double_park_candidate"
                elif cd.bike_encroachment:
                    color_key = "bike_encroachment"
                elif cd.curb_occupancy:
                    color_key = "curb_occupancy"
                elif cd.travel_occupancy:
                    color_key = "travel_occupancy"
                else:
                    color_key = "curb_occupancy"
                color_bgr = _rgb_to_bgr(BEHAVIOR_COLORS_RGB.get(color_key, (255, 255, 255)))
            else:
                # Category-based coloring for non-zone objects
                color_rgb = CATEGORY_COLORS_RGB.get(det.category, (200, 200, 200))
                color_bgr = _rgb_to_bgr(color_rgb)

            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color_bgr, 2)

            if draw_labels:
                label_text = f"{det.label} {det.confidence:.0%}"
                if cd.double_park_candidate:
                    label_text += " DOUBLE-PARK"
                elif cd.bike_encroachment:
                    label_text += " BIKE-BLOCK"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), color_bgr, -1)
                cv2.putText(canvas, label_text, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Summary text in top-left corner
    lines = [
        f"Frame {result.frame_index} | {result.timestamp_sec:.1f}s",
        f"Objects: {result.total_detections}  Vehicles: {result.total_vehicles}"
        f"  Peds: {result.category_counts.get('pedestrian', 0)}",
        f"Curb: {result.curb_occupied_count}  Travel: {result.travel_occupied_count}"
        f"  Bike-block: {result.bike_encroachment_count}  Dbl-park: {result.double_park_count}",
    ]
    y0 = 25
    for line in lines:
        cv2.putText(canvas, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y0 += 22

    return canvas
