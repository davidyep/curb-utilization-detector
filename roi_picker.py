import argparse
import json
from pathlib import Path

import cv2

WINDOW_NAME = "ROI Picker"


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive ROI polygon picker.")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out", default="roi_polygons.json", help="Output JSON path")
    parser.add_argument(
        "--labels",
        default="curb,travel,bike",
        help="Comma-separated ROI labels (default: curb,travel,bike)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(img_path)

    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    if not labels:
        raise ValueError("No ROI labels provided.")

    image = cv2.imread(str(img_path))
    if image is None:
        raise RuntimeError(f"Failed to load image: {img_path}")

    state = {
        "points": [],
        "roi_index": 0,
        "rois": {},
    }

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["points"].append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if state["points"]:
                state["points"].pop()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    instructions = [
        "LMB: add point",
        "RMB: undo last",
        "N: next ROI",
        "C: clear points",
        "S: skip ROI",
        "Q: quit",
    ]

    while state["roi_index"] < len(labels):
        label = labels[state["roi_index"]]
        while True:
            canvas = image.copy()
            # Draw points and edges
            if state["points"]:
                for p in state["points"]:
                    cv2.circle(canvas, p, 4, (0, 255, 255), -1)
                for i in range(1, len(state["points"])):
                    cv2.line(canvas, state["points"][i - 1], state["points"][i], (0, 255, 0), 2)
            # Draw label
            cv2.putText(
                canvas,
                f"ROI: {label}  (points: {len(state['points'])})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            y0 = 60
            for i, line in enumerate(instructions):
                cv2.putText(
                    canvas,
                    line,
                    (10, y0 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                print("Quit without saving remaining ROIs.")
                return
            if key in (ord("c"),):
                state["points"] = []
            if key in (ord("s"),):
                state["points"] = []
                state["roi_index"] += 1
                break
            if key in (ord("n"),):
                if len(state["points"]) < 3:
                    print("Need at least 3 points to form a polygon.")
                    continue
                state["rois"][label] = state["points"]
                state["points"] = []
                state["roi_index"] += 1
                break

    cv2.destroyAllWindows()

    out_path = Path(args.out)
    out_path.write_text(json.dumps(state["rois"], indent=2), encoding="utf-8")
    print(f"Saved ROIs to {out_path.resolve()}")


if __name__ == "__main__":
    main()
