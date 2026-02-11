# NYC Street Scene & Curb Utilization Detector

A Streamlit web app that uses YOLOv8 to analyze NYC street camera feeds for curb utilization, double-parking, bike lane obstructions, pedestrian activity, and more.

## What It Does

Upload an image or video of a street scene, define ROI zones (curb, travel lane, bike lane), and the app detects **30+ object types** across 7 categories:

| Category | Objects Detected |
|---|---|
| **Vehicle** | car, motorcycle, bus, truck |
| **Pedestrian** | person |
| **Cyclist** | bicycle |
| **Street Infrastructure** | traffic light, fire hydrant, stop sign, parking meter, bench |
| **Animal** | bird, cat, dog |
| **Personal Item** | backpack, umbrella, handbag, suitcase |
| **Street Furniture** | chair, potted plant, dining table |

### Key Analytics

- **Curb occupancy** &mdash; vehicles/objects parked in the curb zone
- **Double-parking detection** &mdash; vehicles straddling curb and travel lanes
- **Bike lane obstructions** &mdash; non-cyclist objects blocking the bike lane
- **Pedestrian density** &mdash; pedestrian counts by zone (curb, travel, bike)
- **Category breakdowns** &mdash; per-frame counts for all 7 object categories
- **Time-series analysis** &mdash; zone occupancy and category counts over video duration

## Setup

### Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
streamlit run curb_app.py
```

## Usage

### 1. Define ROI Zones

Before analyzing, you need to define three polygon zones on your camera view:

- **Curb** &mdash; the curbside parking area
- **Travel** &mdash; the active travel lane
- **Bike** &mdash; the bike lane

Three options to create zones:

- **ROI Picker (recommended for first time):** Extract a frame from your video, then run:
  ```bash
  python roi_picker.py --image frame.jpg --out roi_polygons.json
  ```
  Left-click to add points, press N to save and move to next zone, Q to quit.

- **Upload JSON** in the app's ROI Setup tab

- **Draw in browser** using the app's built-in canvas (requires `pip install streamlit-drawable-canvas`)

### 2. Analyze

1. Open the **Analyze** tab
2. Upload an image or video (or provide a file path)
3. Load your ROI zones
4. Click **Analyze**

### 3. Review Results

The dashboard shows:
- Metric cards (detections/frame, vehicles, double-parking, curb utilization, pedestrians, bike lane obstructions)
- Zone occupancy over time (line chart)
- Category counts over time (line chart)
- Category and object type breakdowns (bar charts)
- Filterable detections table (by category, object type, double-park flag)
- Downloadable CSV/JSON exports

## Configuration

All defaults are in `curb_config.py`:

| Setting | Default | Description |
|---|---|---|
| `YOLO_MODEL_NAME` | `yolov8n.pt` | YOLO model (nano/small/medium) |
| `YOLO_CONFIDENCE` | `0.35` | Minimum detection confidence |
| `OVERLAP_THRESHOLD` | `0.20` | Minimum bbox/zone overlap to count as occupied |
| `DEFAULT_FRAME_SKIP` | `30` | Process every Nth frame (~1 fps at 30fps) |
| `DEFAULT_BATCH_SIZE` | `8` | Frames per YOLO inference batch |

The sidebar also exposes these as interactive sliders, along with category toggles to enable/disable detection of specific object types.

## Project Structure

```
curb_app.py          # Streamlit web UI (3 tabs: Analyze, ROI Setup, Results)
curb_detection.py    # Core detection: YOLOv8 wrapper, ROI overlap classification, visualization
curb_video.py        # Video pipeline: batch processing, DataFrames, summary stats, export
curb_config.py       # Constants: COCO taxonomy, colors, thresholds, paths
roi_picker.py        # Interactive OpenCV polygon picker for defining ROI zones
roi_polygons.json    # Example ROI polygon definitions
requirements.txt     # Python dependencies
```

## How It Works

1. **YOLOv8** runs object detection on each frame (all 30+ street-relevant COCO classes)
2. Each detection is assigned a **category** (vehicle, pedestrian, cyclist, etc.)
3. Bounding boxes are checked for **overlap** against the three ROI zone masks
4. Zone-occupancy flags are set for relevant categories (vehicles, cyclists, street furniture):
   - `curb_occupancy` &mdash; object overlaps curb zone
   - `travel_occupancy` &mdash; object overlaps travel lane
   - `bike_encroachment` &mdash; object overlaps bike lane
   - `double_park_candidate` &mdash; vehicle overlaps both curb and travel
5. Per-frame results are aggregated into time-series DataFrames and summary statistics
