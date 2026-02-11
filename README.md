# NYC Street Scene & Curb Utilization Detector

A Streamlit web app that uses YOLOv8 + YOLO-World to analyze NYC street camera feeds for curb utilization, double-parking, bike lane obstructions, pedestrian activity, and street infrastructure.

## What It Does

Upload an image or video of a street scene, define ROI zones (curb, travel lane, bike lane), and the app runs **dual-model detection**:

### YOLO COCO Detection (30+ object types)

| Category | Objects Detected |
|---|---|
| **Vehicle** | car, motorcycle, bus, train, truck |
| **Pedestrian** | person |
| **Cyclist** | bicycle, skateboard |
| **Street Infrastructure** | traffic light, fire hydrant, stop sign, parking meter, bench |
| **Animal** | bird, cat, dog |
| **Personal Item** | backpack, umbrella, handbag, suitcase |
| **Street Furniture** | chair, potted plant, dining table |

### YOLO-World Open-Vocabulary Detection (20+ infrastructure types)

Detects objects that standard COCO models cannot, using text-prompt-based detection:

| Category | Objects Detected |
|---|---|
| **Road Infrastructure** | bollard, traffic cone, jersey barrier, barricade, bike rack, bus shelter, bus stop sign, mailbox, trash can, recycling bin, street light, utility pole, manhole cover |
| **Road Marking** | bike lane, crosswalk |
| **Construction** | scaffolding, construction barrier, construction fence |
| **Vendor** | food cart, food truck |
| **Waste** | dumpster, garbage bag |

### Key Analytics

- **Curb occupancy** &mdash; vehicles/objects parked in the curb zone
- **Double-parking detection** &mdash; vehicles straddling curb and travel lanes
- **Bike lane obstructions** &mdash; non-cyclist objects blocking the bike lane
- **Pedestrian density** &mdash; pedestrian counts by zone (curb, travel, bike)
- **Infrastructure mapping** &mdash; bollards, barriers, construction, vendor carts
- **Category breakdowns** &mdash; per-frame counts for all 12 object categories
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
4. Toggle **YOLO-World** on/off in the sidebar for infrastructure detection
5. Click **Analyze**

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
| `YOLO_MODEL_NAME` | `yolov8s.pt` | COCO model (small/medium/nano) |
| `YOLO_CONFIDENCE` | `0.30` | COCO detection confidence |
| `YOLO_WORLD_MODEL_NAME` | `yolov8s-worldv2.pt` | YOLO-World model |
| `YOLO_WORLD_CONFIDENCE` | `0.15` | Open-vocab confidence (lower = more detections) |
| `OVERLAP_THRESHOLD` | `0.20` | Minimum bbox/zone overlap to count as occupied |
| `DEFAULT_FRAME_SKIP` | `30` | Process every Nth frame (~1 fps at 30fps) |
| `DEFAULT_BATCH_SIZE` | `8` | Frames per inference batch |

The sidebar exposes these as interactive sliders, along with category toggles to enable/disable specific object types from both models.

## Project Structure

```
curb_app.py          # Streamlit web UI (3 tabs: Analyze, ROI Setup, Results)
curb_detection.py    # Core detection: YOLO + YOLO-World, ROI overlap, visualization
curb_video.py        # Video pipeline: batch processing, DataFrames, summary stats, export
curb_config.py       # Constants: COCO + YOLO-World taxonomy, colors, thresholds
roi_picker.py        # Interactive OpenCV polygon picker for defining ROI zones
roi_polygons.json    # Example ROI polygon definitions
requirements.txt     # Python dependencies
```

## How It Works

1. **YOLOv8 COCO** detects standard objects (vehicles, pedestrians, cyclists, etc.)
2. **YOLO-World** detects open-vocabulary infrastructure (bollards, bike lanes, scaffolding, food carts, etc.)
3. Results from both models are **merged** into a unified detection list
4. Each detection is assigned a **category** and checked for **overlap** against ROI zone masks
5. Zone-occupancy flags are set for relevant categories:
   - `curb_occupancy` &mdash; object overlaps curb zone
   - `travel_occupancy` &mdash; object overlaps travel lane
   - `bike_encroachment` &mdash; object overlaps bike lane
   - `double_park_candidate` &mdash; vehicle overlaps both curb and travel
6. Per-frame results are aggregated into time-series DataFrames and summary statistics
