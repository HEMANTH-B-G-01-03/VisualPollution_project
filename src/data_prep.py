# # src/data_prep.py
# import os
# import pandas as pd
# import cv2
# from pathlib import Path

# DATA_DIR   = Path('data')
# IMG_DIR    = DATA_DIR / 'images'
# OUT_LABELS = DATA_DIR / 'labels_yolo'
# CROPS_DIR  = Path('crops')

# OUT_LABELS.mkdir(parents=True, exist_ok=True)
# CROPS_DIR.mkdir(parents=True, exist_ok=True)

# # >>> Set these to match your CSV header exactly <<<
# IMAGE_COL = 'image_path'   # column that holds the image filename or relative path
# CLASS_COL = 'class'        # column with the class label (text)
# # The CSV has columns named xmax,xmin,ymax,ymin (we will use them accordingly)
# XMIN_COL  = 'xmin'
# YMIN_COL  = 'ymin'
# XMAX_COL  = 'xmax'
# YMAX_COL  = 'ymax'
# # >>> end config <<<

# # Update/replace this CLASS_MAP so it matches all distinct class names present in your CSV.
# # If you don't know them yet, run the small check below to see unique labels in the CSV,
# # then fill CLASS_MAP accordingly.
# CLASS_MAP = {
#     'pothole':0, 'garbage':1, 'barrier':2, 'damaged_footpath':3,
#     'construction_waste':4, 'abandoned_vehicle':5, 'uneven_road':6, 'other':7
# }

# def safe_int(x):
#     try:
#         return int(float(x))
#     except Exception:
#         return None

# def convert_to_yolo(row):
#     fname = str(row[IMAGE_COL])
#     # if image path contains folders, take them as relative to data/images
#     img_path = IMG_DIR / Path(fname).name
#     img = cv2.imread(str(img_path))
#     if img is None:
#         print(f"[WARN] Could not read image {img_path}")
#         return None
#     h, w = img.shape[:2]

#     xmin = safe_int(row[XMIN_COL])
#     ymin = safe_int(row[YMIN_COL])
#     xmax = safe_int(row[XMAX_COL])
#     ymax = safe_int(row[YMAX_COL])

#     if None in (xmin, ymin, xmax, ymax):
#         print(f"[WARN] Invalid bbox for {fname}: {xmin},{ymin},{xmax},{ymax}")
#         return None

#     # clamp values
#     xmin = max(0, min(xmin, w-1))
#     xmax = max(0, min(xmax, w-1))
#     ymin = max(0, min(ymin, h-1))
#     ymax = max(0, min(ymax, h-1))
#     if xmax <= xmin or ymax <= ymin:
#         print(f"[WARN] Empty/invalid bbox for {fname}: xmin>{xmax} or ymin>{ymax}")
#         return None

#     x_center = (xmin + xmax) / 2.0 / w
#     y_center = (ymin + ymax) / 2.0 / h
#     bw = (xmax - xmin) / w
#     bh = (ymax - ymin) / h

#     cls_name = str(row[CLASS_COL])
#     if cls_name not in CLASS_MAP:
#         print(f"[WARN] Class '{cls_name}' not in CLASS_MAP — skipping (image={fname})")
#         return None

#     cls = CLASS_MAP[cls_name]
#     return f"{cls} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}", img, xmin, ymin, xmax, ymax, cls

# def main(csv_path='data/train.csv'):
#     print("Reading:", csv_path)
#     df = pd.read_csv(csv_path)
#     print("CSV columns:", list(df.columns))
#     print("Sample rows:")
#     print(df.head(3))

#     # quick check of unique classes
#     unique_classes = df[CLASS_COL].unique() if CLASS_COL in df.columns else []
#     print("Unique classes in CSV:", unique_classes)
#     missing = [c for c in unique_classes if c not in CLASS_MAP]
#     if missing:
#         print("[NOTICE] The following classes are missing from CLASS_MAP:", missing)
#         print("[ACTION] Edit src/data_prep.py CLASS_MAP to include them BEFORE proceeding.")
#         # we will still try to process rows but skip unknown classes

#     count = 0
#     for fname, group in df.groupby(IMAGE_COL):
#         txt_path = OUT_LABELS / (Path(fname).stem + '.txt')
#         lines = []
#         for idx, r in group.iterrows():
#             res = convert_to_yolo(r)
#             if res is None:
#                 continue
#             line, img, xmin, ymin, xmax, ymax, cls = res
#             lines.append(line)
#             # save crop for classifier training
#             crop = img[ymin:ymax, xmin:xmax]
#             crop_name = f"{Path(fname).stem}_{idx}_{cls}.jpg"
#             cv2.imwrite(str(CROPS_DIR / crop_name), crop)
#             count += 1
#         if lines:
#             txt_path.write_text("\n".join(lines))
#     print(f"Done. Created label files in: {OUT_LABELS} and {count} crop images in: {CROPS_DIR}")

# if __name__ == "__main__":
#     main()




# src/data_prep.py
import os
import pandas as pd
import cv2
from pathlib import Path
import json

DATA_DIR   = Path('data')
IMG_DIR    = DATA_DIR / 'images'
OUT_LABELS = DATA_DIR / 'labels_yolo'
CROPS_DIR  = Path('crops')

OUT_LABELS.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

# CSV column names (your header: class,image_path,name,xmax,xmin,ymax,ymin)
IMAGE_COL = 'image_path'
CLASS_COL = 'class'
NAME_COL  = 'name'   # textual name column if present (optional)
XMIN_COL  = 'xmin'
YMIN_COL  = 'ymin'
XMAX_COL  = 'xmax'
YMAX_COL  = 'ymax'

def safe_int(x):
    try:
        return int(float(x))
    except Exception:
        return None

def read_df(csv_path='data/train.csv'):
    df = pd.read_csv(csv_path)
    return df

def build_class_maps(df):
    # Use NAME_COL if present to preserve textual class names
    class_vals = list(df[CLASS_COL].unique())
    # Try to build a mapping from original class value -> text name
    id_to_name = {}
    if NAME_COL in df.columns:
        # For each unique class val, pick first corresponding 'name' value
        for v in class_vals:
            row = df[df[CLASS_COL] == v].iloc[0]
            id_to_name[str(v)] = str(row[NAME_COL])
    else:
        # fallback: use numeric/text value to create a readable name
        for v in class_vals:
            id_to_name[str(v)] = "class_" + str(v).replace('.0','')
    # Now sort class keys in a stable numeric-aware order
    def keyfun(k):
        try:
            return float(k)
        except:
            return k
    sorted_keys = sorted(id_to_name.keys(), key=keyfun)
    # Create final maps: original_value(str) -> new_index (0..n-1), and index->name
    orig_to_index = {}
    index_to_name = []
    for i, orig in enumerate(sorted_keys):
        orig_to_index[orig] = i
        index_to_name.append(id_to_name[orig])
    return orig_to_index, index_to_name

def convert_to_yolo_and_crop(row, orig_to_index):
    fname = str(row[IMAGE_COL])
    img_path = IMG_DIR / Path(fname).name
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] Could not read image {img_path}")
        return None
    h, w = img.shape[:2]

    xmin = safe_int(row[XMIN_COL])
    ymin = safe_int(row[YMIN_COL])
    xmax = safe_int(row[XMAX_COL])
    ymax = safe_int(row[YMAX_COL])

    if None in (xmin, ymin, xmax, ymax):
        print(f"[WARN] Invalid bbox for {fname}: {xmin},{ymin},{xmax},{ymax}")
        return None

    xmin = max(0, min(xmin, w-1))
    xmax = max(0, min(xmax, w-1))
    ymin = max(0, min(ymin, h-1))
    ymax = max(0, min(ymax, h-1))
    if xmax <= xmin or ymax <= ymin:
        print(f"[WARN] Empty/invalid bbox for {fname}: xmin>{xmax} or ymin>{ymax}")
        return None

    x_center = (xmin + xmax) / 2.0 / w
    y_center = (ymin + ymax) / 2.0 / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h

    orig_val = str(row[CLASS_COL])
    if orig_val not in orig_to_index:
        print(f"[WARN] Class '{orig_val}' not found in mapping — skipping (image={fname})")
        return None
    cls_idx = orig_to_index[orig_val]
    yolo_line = f"{cls_idx} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}"
    return yolo_line, img, xmin, ymin, xmax, ymax, cls_idx

def main(csv_path='data/train.csv'):
    print("Reading", csv_path)
    df = read_df(csv_path)
    print("CSV columns:", list(df.columns))
    # Build mapping
    orig_to_index, index_to_name = build_class_maps(df)
    print("Detected classes (original -> new index):")
    for orig, idx in orig_to_index.items():
        print(f"  {orig} -> {idx}  (name: {index_to_name[idx]})")
    # write names to data/names.txt for convenience (and for data.yaml)
    names_file = DATA_DIR / 'names.txt'
    with open(names_file, 'w', encoding='utf-8') as f:
        for n in index_to_name:
            f.write(n + "\n")
    print(f"Wrote class names to: {names_file}")

    count = 0
    for fname, group in df.groupby(IMAGE_COL):
        txt_path = OUT_LABELS / (Path(fname).stem + '.txt')
        lines = []
        for idx, r in group.iterrows():
            res = convert_to_yolo_and_crop(r, orig_to_index)
            if res is None:
                continue
            line, img, xmin, ymin, xmax, ymax, cls = res
            lines.append(line)
            crop = img[ymin:ymax, xmin:xmax]
            crop_name = f"{Path(fname).stem}_{idx}_{cls}.jpg"
            cv2.imwrite(str(CROPS_DIR / crop_name), crop)
            count += 1
        if lines:
            txt_path.write_text("\n".join(lines))
    print(f"Done. Created labels in {OUT_LABELS} and {count} crops in {CROPS_DIR}")

if __name__ == "__main__":
    main()
