# src/detect_and_classify.py
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from preprocess import preprocess_image
   # relative import within src/

# ---- Paths (adjust if you put files elsewhere) ----
YOLO_WEIGHTS = Path('models/yolov5_best.pt')   # copy your best.pt -> models/yolov5_best.pt
NAMES_FILE   = Path('data/names.txt')         # created by data_prep.py

# ---- Inference settings ----
CONF_THRESH = 0.25
IMGSZ = 640

# Load class names
if NAMES_FILE.exists():
    CLASS_NAMES = [x.strip() for x in NAMES_FILE.read_text(encoding='utf-8').splitlines() if x.strip()]
else:
    # fallback if names.txt missing
    CLASS_NAMES = [f"class_{i}" for i in range(100)]

# Load model
if not YOLO_WEIGHTS.exists():
    raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS}. Please copy trained best.pt to models/yolov5_best.pt")
yolo = YOLO(str(YOLO_WEIGHTS))

def detect_and_annotate(input_path, out_path='out.jpg', conf=CONF_THRESH):
    """Run YOLO on input image, draw green boxes + labels, save and return out_path."""
    img_bgr = cv2.imread(str(input_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read input image: {input_path}")

    # Preprocess same as earlier
    img_proc = preprocess_image(img_bgr)

    # Run detection (ultralytics)
    results = yolo.predict(source=img_proc, conf=conf, imgsz=IMGSZ)
    if len(results) == 0:
        cv2.imwrite(out_path, img_bgr)
        return out_path

    res = results[0]
    # boxes, confidences and class indices
    boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes, 'xyxy') else np.empty((0,4))
    confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, 'conf') else np.zeros(len(boxes))
    cls_inds = res.boxes.cls.cpu().numpy().astype(int) if hasattr(res.boxes, 'cls') else np.zeros(len(boxes), dtype=int)

    h, w = img_bgr.shape[:2]
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        x1 = int(max(0, np.floor(x1))); y1 = int(max(0, np.floor(y1)))
        x2 = int(min(w-1, np.ceil(x2)));  y2 = int(min(h-1, np.ceil(y2)))
        cls_idx = int(cls_inds[i]) if i < len(cls_inds) else -1
        label = CLASS_NAMES[cls_idx] if 0 <= cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"
        score = float(confs[i]) if i < len(confs) else 0.0
        text = f"{label} {score:.2f}"

        # Draw rectangle (green)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0,255,0), 2)

        # Draw filled rectangle behind text for readability
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(0, y1 - th - 6)
        cv2.rectangle(img_bgr, (x1, y_text), (x1 + tw + 6, y1), (0,255,0), -1)
        cv2.putText(img_bgr, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, img_bgr)
    return out_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/detect_and_classify.py input.jpg [out.jpg]")
    else:
        inp = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) > 2 else 'out.jpg'
        detect_and_annotate(inp, out)
        print("Saved ->", out)
