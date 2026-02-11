"""Shared constants for the curb utilization detection system."""

from pathlib import Path

# --------------- COCO Class Taxonomy for NYC Streets --------------- #

# Every COCO class ID relevant to NYC street scenes, grouped by category.
COCO_CATEGORIES: dict[str, dict[int, str]] = {
    "vehicle": {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    },
    "pedestrian": {
        0: "person",
    },
    "cyclist": {
        1: "bicycle",
    },
    "street_infrastructure": {
        9: "traffic light",
        10: "fire hydrant",
        11: "stop sign",
        12: "parking meter",
        13: "bench",
    },
    "animal": {
        14: "bird",
        15: "cat",
        16: "dog",
    },
    "personal_item": {
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        28: "suitcase",
    },
    "street_furniture": {
        56: "chair",
        58: "potted plant",
        60: "dining table",
    },
}

# Flat lookup: class_id -> (category, label)
COCO_CLASS_LOOKUP: dict[int, tuple[str, str]] = {
    cls_id: (cat_name, label)
    for cat_name, classes in COCO_CATEGORIES.items()
    for cls_id, label in classes.items()
}

# All tracked class IDs (union of all categories)
ALL_TRACKED_CLASS_IDS: set[int] = set(COCO_CLASS_LOOKUP.keys())

# Backward-compatible alias
VEHICLE_CLASS_IDS: dict[int, str] = COCO_CATEGORIES["vehicle"]

# Which categories get curb/travel/bike zone-occupancy flags
ZONE_ANALYSIS_CATEGORIES: set[str] = {"vehicle", "cyclist", "street_furniture"}

# Which categories count as potential obstructions (e.g. in bike lane)
OBSTRUCTION_CATEGORIES: set[str] = {"vehicle", "cyclist", "street_furniture", "personal_item"}

# --------------- YOLO ---------------
YOLO_MODEL_NAME: str = "yolov8n.pt"  # nano; swap to yolov8s.pt / yolov8m.pt for accuracy
YOLO_CONFIDENCE: float = 0.35
YOLO_IOU_NMS: float = 0.45

# --------------- ROI / Overlap ---------------
DEFAULT_ROI_LABELS: list[str] = ["curb", "travel", "bike"]
OVERLAP_THRESHOLD: float = 0.20

# --------------- Video Processing ---------------
DEFAULT_FRAME_SKIP: int = 30  # process every Nth frame (~1 fps at 30 fps video)
DEFAULT_BATCH_SIZE: int = 8

# --------------- Paths ---------------
PROJECT_DIR: Path = Path(__file__).resolve().parent
DEFAULT_ROI_PATH: Path = PROJECT_DIR / "roi_polygons.json"
DEFAULT_OUTPUT_DIR: Path = PROJECT_DIR / "curb_output"

# --------------- Visualization Colors (RGB) ---------------
ZONE_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "curb": (65, 105, 225),   # royal blue
    "travel": (220, 20, 60),  # crimson
    "bike": (34, 139, 34),    # forest green
}

BEHAVIOR_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "curb_occupancy": (65, 105, 225),
    "travel_occupancy": (220, 20, 60),
    "bike_encroachment": (34, 139, 34),
    "double_park_candidate": (255, 165, 0),  # orange
}

CATEGORY_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "vehicle": (255, 165, 0),          # orange
    "pedestrian": (0, 191, 255),       # deep sky blue
    "cyclist": (50, 205, 50),          # lime green
    "street_infrastructure": (169, 169, 169),  # dark gray
    "animal": (255, 105, 180),         # hot pink
    "personal_item": (148, 103, 189),  # medium purple
    "street_furniture": (139, 90, 43), # saddle brown
}
