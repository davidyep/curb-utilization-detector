"""
Streamlit web UI for NYC street-scene curb utilization detection.

Run:  streamlit run curb_app.py
"""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from curb_config import (
    COCO_CATEGORIES,
    WORLD_CATEGORIES,
    CATEGORY_COLORS_RGB,
    DEFAULT_ROI_LABELS,
    OVERLAP_THRESHOLD,
    DEFAULT_FRAME_SKIP,
    DEFAULT_BATCH_SIZE,
    YOLO_CONFIDENCE,
    YOLO_IOU_NMS,
    YOLO_MODEL_NAME,
    YOLO_WORLD_CONFIDENCE,
    ZONE_COLORS_RGB,
    BEHAVIOR_COLORS_RGB,
    DEFAULT_OUTPUT_DIR,
)
from curb_detection import (
    StreetSceneDetector,
    InfrastructureDetector,
    CombinedDetector,
    FrameResult,
    load_roi_polygons,
    build_roi_masks,
    draw_annotated_frame,
)
from curb_video import (
    process_video,
    process_image,
    get_video_metadata,
    results_to_dataframe,
    detections_to_dataframe,
    compute_summary_stats,
    export_results,
)

# ------------------------------------------------------------------ #
#  Page Config
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="NYC Street Scene Analyzer",
    layout="wide",
)

# ------------------------------------------------------------------ #
#  Sidebar: Parameters
# ------------------------------------------------------------------ #

st.sidebar.title("Settings")

model_name = st.sidebar.selectbox(
    "YOLO Model",
    ["yolov8s.pt", "yolov8m.pt", "yolov8n.pt"],
    index=0,
    help="small = good balance, medium = most accurate, nano = fastest",
)
yolo_conf = st.sidebar.slider(
    "Detection Confidence", 0.10, 0.90, YOLO_CONFIDENCE, 0.05,
)
overlap_thresh = st.sidebar.slider(
    "Overlap Threshold", 0.05, 0.50, OVERLAP_THRESHOLD, 0.05,
    help="Minimum bbox/zone overlap to count as occupied",
)
frame_skip = st.sidebar.slider(
    "Frame Skip (video)", 1, 120, DEFAULT_FRAME_SKIP, 1,
    help="Process every Nth frame. 30 = ~1 fps for 30fps video",
)
batch_size = st.sidebar.slider(
    "Batch Size", 1, 32, DEFAULT_BATCH_SIZE, 1,
)

# --- YOLO-World toggle ---
st.sidebar.divider()
enable_world = st.sidebar.toggle(
    "YOLO-World (infrastructure)",
    value=True,
    help="Detect bollards, bike lanes, scaffolding, food carts, dumpsters, etc.",
)
world_conf = st.sidebar.slider(
    "World Confidence", 0.05, 0.50, YOLO_WORLD_CONFIDENCE, 0.05,
    help="Open-vocab detection threshold (lower = more detections)",
) if enable_world else YOLO_WORLD_CONFIDENCE

# --- Category Toggles ---
st.sidebar.divider()
st.sidebar.subheader("Detection Categories")

st.sidebar.caption("COCO classes")
selected_categories: list[str] = []
for cat_name in COCO_CATEGORIES.keys():
    if st.sidebar.checkbox(
        cat_name.replace("_", " ").title(),
        value=True,
        key=f"cat_{cat_name}",
    ):
        selected_categories.append(cat_name)

if enable_world:
    st.sidebar.caption("YOLO-World classes")
    selected_world_categories: list[str] = []
    for cat_name in WORLD_CATEGORIES.keys():
        if st.sidebar.checkbox(
            cat_name.replace("_", " ").title(),
            value=True,
            key=f"wcat_{cat_name}",
        ):
            selected_world_categories.append(cat_name)
else:
    selected_world_categories = []

# ------------------------------------------------------------------ #
#  Detector Cache
# ------------------------------------------------------------------ #

@st.cache_resource
def get_coco_detector(
    mn: str, conf: float, iou: float, cats: frozenset | None = None,
) -> StreetSceneDetector:
    return StreetSceneDetector(
        model_name=mn,
        confidence=conf,
        iou_threshold=iou,
        categories=set(cats) if cats else None,
    )


@st.cache_resource
def get_world_detector(conf: float, iou: float) -> InfrastructureDetector:
    return InfrastructureDetector(confidence=conf, iou_threshold=iou)


def build_detector(
    mn: str, conf: float, iou: float,
    cats: frozenset | None, use_world: bool, world_conf: float,
):
    """Build the appropriate detector based on settings."""
    coco = get_coco_detector(mn, conf, iou, cats)
    if use_world:
        world = get_world_detector(world_conf, iou)
        return CombinedDetector(coco, world)
    return coco


# ------------------------------------------------------------------ #
#  Helper: load ROI from various sources
# ------------------------------------------------------------------ #

def _load_roi_from_upload(uploaded_file) -> dict[str, list[tuple[int, int]]]:
    raw = json.loads(uploaded_file.read().decode("utf-8"))
    polygons = {}
    for name, pts in raw.items():
        if len(pts) >= 3:
            polygons[name] = [(int(x), int(y)) for x, y in pts]
    return polygons


def _frame_with_roi_overlay(frame_bgr: np.ndarray, roi_polys: dict) -> np.ndarray:
    """Draw ROI polygons on a frame and return RGB for Streamlit."""
    canvas = frame_bgr.copy()
    overlay = canvas.copy()
    for name, pts in roi_polys.items():
        color_rgb = ZONE_COLORS_RGB.get(name, (128, 128, 128))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [arr], color_bgr)
        cv2.polylines(canvas, [arr], True, color_bgr, 2)
        # Label
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
        cv2.putText(canvas, name, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ #
#  Tabs
# ------------------------------------------------------------------ #

tab_analyze, tab_roi, tab_results = st.tabs(["Analyze", "ROI Setup", "Results"])


# ================================================================== #
#  TAB 1: ANALYZE
# ================================================================== #

with tab_analyze:
    st.header("Street Scene & Curb Utilization Analysis")

    # --- Upload Section ---
    col_upload, col_roi = st.columns(2)

    with col_upload:
        st.subheader("1. Upload Image or Video")
        upload_mode = st.radio(
            "Input method", ["Upload file", "Path on disk"], horizontal=True,
            label_visibility="collapsed",
        )
        media_file = None
        media_path = None
        is_video = False

        if upload_mode == "Upload file":
            media_file = st.file_uploader(
                "Choose an image or video",
                type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"],
            )
            if media_file is not None:
                ext = Path(media_file.name).suffix.lower()
                is_video = ext in {".mp4", ".avi", ".mov", ".mkv"}
        else:
            path_input = st.text_input("Full file path", placeholder=r"C:\Videos\street_cam.mp4")
            if path_input and Path(path_input).exists():
                media_path = path_input
                ext = Path(path_input).suffix.lower()
                is_video = ext in {".mp4", ".avi", ".mov", ".mkv"}
            elif path_input:
                st.warning("File not found.")

    with col_roi:
        st.subheader("2. Load ROI Zones")
        roi_source = st.radio(
            "ROI source",
            ["Upload JSON", "Default (roi_polygons.json)", "From ROI Setup tab"],
            horizontal=True,
            label_visibility="collapsed",
        )
        roi_polys = None

        if roi_source == "Upload JSON":
            roi_file = st.file_uploader("Upload roi_polygons.json", type=["json"], key="roi_upload")
            if roi_file is not None:
                try:
                    roi_polys = _load_roi_from_upload(roi_file)
                    st.success(f"Loaded {len(roi_polys)} zones: {', '.join(roi_polys.keys())}")
                except Exception as e:
                    st.error(f"Invalid ROI JSON: {e}")
        elif roi_source == "Default (roi_polygons.json)":
            try:
                roi_polys = load_roi_polygons()
                st.success(f"Loaded {len(roi_polys)} zones: {', '.join(roi_polys.keys())}")
            except FileNotFoundError:
                st.warning("roi_polygons.json not found. Use ROI Setup tab or upload one.")
        else:
            if "roi_polygons" in st.session_state and st.session_state["roi_polygons"]:
                roi_polys = st.session_state["roi_polygons"]
                st.success(f"Using {len(roi_polys)} zones from ROI Setup tab.")
            else:
                st.info("Draw zones in the ROI Setup tab first.")

    # --- Preview ---
    if roi_polys and (media_file is not None or media_path is not None):
        with st.expander("Preview: first frame with ROI overlay", expanded=False):
            if media_path:
                if is_video:
                    cap = cv2.VideoCapture(media_path)
                    ret, preview_frame = cap.read()
                    cap.release()
                else:
                    preview_frame = cv2.imread(media_path)
                    ret = preview_frame is not None
            else:
                # Write uploaded file to temp for OpenCV
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(media_file.name).suffix)
                tmp.write(media_file.read())
                tmp.flush()
                media_file.seek(0)  # reset for later use
                if is_video:
                    cap = cv2.VideoCapture(tmp.name)
                    ret, preview_frame = cap.read()
                    cap.release()
                else:
                    preview_frame = cv2.imread(tmp.name)
                    ret = preview_frame is not None

            if ret and preview_frame is not None:
                st.image(_frame_with_roi_overlay(preview_frame, roi_polys), use_container_width=True)
            else:
                st.warning("Could not read preview frame.")

    # --- Run Analysis ---
    st.subheader("3. Run Analysis")
    can_run = roi_polys is not None and (media_file is not None or media_path is not None)

    if not can_run:
        st.info("Upload a file and load ROI zones to begin.")

    if can_run and st.button("Analyze", type="primary", use_container_width=True):
        # Resolve the file to a path on disk
        if media_path:
            analysis_path = media_path
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(media_file.name).suffix)
            tmp.write(media_file.read())
            tmp.flush()
            analysis_path = tmp.name

        cats_frozen = frozenset(selected_categories) if selected_categories else None
        detector = build_detector(
            model_name, yolo_conf, YOLO_IOU_NMS,
            cats_frozen, enable_world, world_conf,
        )

        if is_video:
            meta = get_video_metadata(analysis_path)
            st.text(
                f"Video: {meta['width']}x{meta['height']} @ {meta['fps']:.0f}fps | "
                f"{meta['duration_sec']:.0f}s | {meta['total_frames']} frames | "
                f"analyzing ~{meta['total_frames'] // frame_skip} frames"
            )

            progress_bar = st.progress(0)
            status_text = st.empty()

            def _progress(processed: int, total: int, latest: FrameResult):
                frac = min(processed / max(total, 1), 1.0)
                progress_bar.progress(frac)
                status_text.text(
                    f"Frame {processed}/{total} | "
                    f"Detections: {latest.total_detections} | "
                    f"Vehicles: {latest.total_vehicles} | "
                    f"Double-park: {latest.double_park_count}"
                )

            with st.spinner("Running street-scene detection..."):
                results = process_video(
                    analysis_path, roi_polys, detector,
                    frame_skip=frame_skip,
                    batch_size=batch_size,
                    overlap_threshold=overlap_thresh,
                    progress_callback=_progress,
                )

            progress_bar.progress(1.0)
            status_text.text(f"Done! Analyzed {len(results)} frames.")

        else:
            with st.spinner("Detecting objects..."):
                img = cv2.imread(analysis_path)
                roi_masks = build_roi_masks(roi_polys, img.shape[:2])
                from curb_detection import analyze_frame as _af
                single_result = _af(img, detector, roi_masks, overlap_threshold=overlap_thresh)
                results = [single_result]

            # Show annotated image
            annotated = draw_annotated_frame(img, single_result, roi_polys)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Store results in session state
        st.session_state["last_results"] = results

    # Render dashboard if results exist from this or previous run
    if "last_results" in st.session_state and st.session_state["last_results"]:
        results = st.session_state["last_results"]
        stats = compute_summary_stats(results)
        ts_df = results_to_dataframe(results)
        det_df = detections_to_dataframe(results)

        st.divider()
        st.subheader("Results Dashboard")

        # --- Row 1: Core metrics ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Frames Analyzed", stats["total_frames_analyzed"])
        m2.metric("Avg Detections/Frame", stats.get("avg_detections_per_frame", 0))
        m3.metric("Avg Vehicles/Frame", stats["avg_vehicles_per_frame"])
        m4.metric("Double-Park Frames", f"{stats['total_double_park_frames']} ({stats['double_park_pct']}%)")

        # --- Row 2: Curb metrics ---
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Duration (sec)", stats["duration_analyzed_sec"])
        m5.metric("Curb Utilization", f"{stats['curb_utilization_pct']}%")
        m6.metric("Avg Curb Occupancy", stats["avg_curb_occupancy"])
        m6.metric("Peak Curb Occupancy", stats["peak_curb_occupancy"])

        # --- Row 2 cont: Pedestrian & bike lane metrics ---
        ped_stats = stats.get("pedestrian_stats", {})
        bike_stats = stats.get("bike_lane_stats", {})
        m7.metric("Avg Pedestrians/Frame", ped_stats.get("avg_per_frame", 0))
        m7.metric("Peds in Travel Lane (frames)", ped_stats.get("frames_with_pedestrians_in_travel", 0))
        m8.metric("Bike Lane Obstructions", f"{bike_stats.get('obstruction_frames', 0)} ({bike_stats.get('obstruction_pct', 0)}%)")
        m8.metric("Max Detections/Frame", stats.get("max_detections_in_frame", 0))

        # --- Zone occupancy over time ---
        if len(ts_df) > 1:
            st.subheader("Zone Occupancy Over Time")
            zone_cols = ["curb_occupied", "travel_occupied", "bike_encroachment", "double_park"]
            st.line_chart(ts_df.set_index("timestamp_sec")[zone_cols])

            # Category counts over time
            st.subheader("Category Counts Over Time")
            cat_cols = [
                "total_vehicles", "pedestrians", "cyclists",
                "street_infrastructure", "animals", "personal_items", "street_furniture",
                "road_infrastructure", "road_markings", "construction", "vendors", "waste",
            ]
            available = [c for c in cat_cols if c in ts_df.columns and ts_df[c].sum() > 0]
            if available:
                st.line_chart(ts_df.set_index("timestamp_sec")[available])

        # --- Category breakdown ---
        if stats.get("category_breakdown"):
            st.subheader("Detection Category Breakdown")
            cat_df = pd.DataFrame(
                list(stats["category_breakdown"].items()),
                columns=["Category", "Count"],
            ).sort_values("Count", ascending=False)
            st.bar_chart(cat_df.set_index("Category"))

        # --- Object type detail ---
        if stats["vehicle_type_breakdown"]:
            st.subheader("Object Type Detail")
            vtype_df = pd.DataFrame(
                list(stats["vehicle_type_breakdown"].items()),
                columns=["Object Type", "Count"],
            ).sort_values("Count", ascending=False)
            st.bar_chart(vtype_df.set_index("Object Type"))

        # --- Detections table ---
        st.subheader("Detailed Detections")
        if not det_df.empty:
            # Filters
            fcol1, fcol2, fcol3 = st.columns(3)
            with fcol1:
                category_filter = st.multiselect(
                    "Filter by category",
                    det_df["category"].unique().tolist(),
                    default=det_df["category"].unique().tolist(),
                )
            with fcol2:
                filtered_labels = det_df[det_df["category"].isin(category_filter)]["label"].unique().tolist()
                label_filter = st.multiselect(
                    "Filter by object type",
                    filtered_labels,
                    default=filtered_labels,
                )
            with fcol3:
                show_dp_only = st.checkbox("Show double-park candidates only")

            filtered = det_df[
                det_df["category"].isin(category_filter) & det_df["label"].isin(label_filter)
            ]
            if show_dp_only:
                filtered = filtered[filtered["double_park_candidate"]]

            st.dataframe(filtered, use_container_width=True, height=400)
        else:
            st.info("No detections in the analyzed frames.")

        # --- Downloads ---
        st.subheader("Export")
        dcol1, dcol2, dcol3 = st.columns(3)
        with dcol1:
            st.download_button(
                "Download Time-Series CSV",
                ts_df.to_csv(index=False),
                "curb_timeseries.csv",
                "text/csv",
            )
        with dcol2:
            if not det_df.empty:
                st.download_button(
                    "Download Detections CSV",
                    det_df.to_csv(index=False),
                    "curb_detections.csv",
                    "text/csv",
                )
        with dcol3:
            st.download_button(
                "Download Summary JSON",
                json.dumps(stats, indent=2),
                "curb_summary.json",
                "application/json",
            )


# ================================================================== #
#  TAB 2: ROI SETUP
# ================================================================== #

with tab_roi:
    st.header("ROI Zone Setup")
    st.write("Define curb, travel lane, and bike lane zones for your camera view.")

    roi_method = st.radio(
        "Method",
        ["Upload JSON file", "Draw in browser", "Use external roi_picker.py"],
        horizontal=True,
    )

    if roi_method == "Upload JSON file":
        roi_upload = st.file_uploader("Upload roi_polygons.json", type=["json"], key="roi_setup_upload")
        if roi_upload:
            try:
                polys = _load_roi_from_upload(roi_upload)
                st.session_state["roi_polygons"] = polys
                st.success(f"Loaded zones: {', '.join(polys.keys())}")

                # Preview on reference image
                ref_img = st.file_uploader("Upload a reference image to preview", type=["jpg", "png"], key="roi_ref")
                if ref_img:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                    tmp.write(ref_img.read())
                    tmp.flush()
                    frame = cv2.imread(tmp.name)
                    if frame is not None:
                        st.image(_frame_with_roi_overlay(frame, polys), use_container_width=True)
            except Exception as e:
                st.error(str(e))

    elif roi_method == "Draw in browser":
        try:
            from streamlit_drawable_canvas import st_canvas

            st.write("Upload a reference frame, then draw polygons for each zone.")
            ref_upload = st.file_uploader(
                "Reference image", type=["jpg", "jpeg", "png"], key="canvas_ref",
            )

            if ref_upload:
                from PIL import Image
                ref_image = Image.open(ref_upload)
                img_w, img_h = ref_image.size

                current_zone = st.selectbox("Drawing zone", DEFAULT_ROI_LABELS)
                zone_color = ZONE_COLORS_RGB.get(current_zone, (128, 128, 128))
                hex_color = "#{:02x}{:02x}{:02x}".format(*zone_color)

                canvas_result = st_canvas(
                    fill_color=hex_color + "40",  # semi-transparent fill
                    stroke_width=2,
                    stroke_color=hex_color,
                    background_image=ref_image,
                    drawing_mode="polygon",
                    height=img_h,
                    width=img_w,
                    key=f"canvas_{current_zone}",
                )

                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data.get("objects", [])
                    if objects:
                        # Initialize stored polygons
                        if "drawn_roi" not in st.session_state:
                            st.session_state["drawn_roi"] = {}

                        # Extract polygon points from canvas objects
                        for obj in objects:
                            if obj.get("type") == "path":
                                path = obj.get("path", [])
                                pts = []
                                for cmd in path:
                                    if len(cmd) >= 3 and cmd[0] in ("M", "L"):
                                        pts.append((int(cmd[1]), int(cmd[2])))
                                if len(pts) >= 3:
                                    st.session_state["drawn_roi"][current_zone] = pts

                        drawn = st.session_state.get("drawn_roi", {})
                        if drawn:
                            st.write("Zones defined:", list(drawn.keys()))

                        if st.button("Save ROI Polygons"):
                            st.session_state["roi_polygons"] = st.session_state["drawn_roi"]
                            st.success("ROI polygons saved! Switch to the Analyze tab to use them.")
        except ImportError:
            st.warning(
                "streamlit-drawable-canvas is not installed. "
                "Install it with: `pip install streamlit-drawable-canvas`\n\n"
                "In the meantime, use 'Upload JSON file' or 'Use external roi_picker.py'."
            )

    else:  # External roi_picker.py
        st.write("Run the following command in your terminal to draw ROI zones interactively:")
        st.code(
            'python roi_picker.py --image "path/to/reference_frame.jpg" --out roi_polygons.json',
            language="bash",
        )
        st.write(
            "This opens an OpenCV window where you can click to draw polygons "
            "for each zone (curb, travel, bike). When done, upload the resulting "
            "JSON above or select 'Default (roi_polygons.json)' in the Analyze tab."
        )
        st.write("**Tip**: To extract a frame from a video for ROI drawing:")
        st.code(
            'python -c "import cv2; c=cv2.VideoCapture(\'video.mp4\'); _, f=c.read(); '
            'cv2.imwrite(\'frame.jpg\', f); c.release()"',
            language="python",
        )


# ================================================================== #
#  TAB 3: RESULTS HISTORY
# ================================================================== #

with tab_results:
    st.header("Past Results")

    output_dir = DEFAULT_OUTPUT_DIR
    if output_dir.exists():
        csv_files = sorted(output_dir.glob("*_timeseries.csv"), reverse=True)
        if csv_files:
            selected = st.selectbox(
                "Select a past analysis",
                [f.stem.replace("_timeseries", "") for f in csv_files],
            )
            if selected:
                ts_path = output_dir / f"{selected}_timeseries.csv"
                det_path = output_dir / f"{selected}_detections.csv"
                summary_path = output_dir / f"{selected}_summary.json"

                if ts_path.exists():
                    ts = pd.read_csv(ts_path)
                    st.subheader("Time Series")
                    # Use available columns (backward compat with old CSVs)
                    zone_cols = ["curb_occupied", "travel_occupied", "bike_encroachment", "double_park"]
                    available_cols = [c for c in zone_cols if c in ts.columns]
                    if available_cols:
                        st.line_chart(ts.set_index("timestamp_sec")[available_cols])
                    st.dataframe(ts, use_container_width=True)

                if summary_path.exists():
                    st.subheader("Summary")
                    st.json(json.loads(summary_path.read_text(encoding="utf-8")))

                if det_path.exists():
                    st.subheader("Detections")
                    st.dataframe(pd.read_csv(det_path), use_container_width=True, height=400)
        else:
            st.info("No past analyses found. Run an analysis first, then export results.")
    else:
        st.info(f"Output directory not found: {output_dir}")

    # Manual export button
    if "last_results" in st.session_state and st.session_state["last_results"]:
        st.divider()
        if st.button("Save current results to disk"):
            paths = export_results(st.session_state["last_results"])
            st.success(f"Saved to: {', '.join(str(p) for p in paths.values())}")
