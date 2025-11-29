# src/data_prep.py
import os, pandas as pd, cv2
from pathlib import Path

DATA_DIR = Path('data')
IMG_DIR = DATA_DIR/'images'
OUT_LABELS = DATA_DIR/'labels_yolo'
CROPS_DIR = Path('crops')

OUT_LABELS.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

# Map your textual classes to integers (modify to match your dataset's class names)
CLASS_MAP = {
    'pothole':0, 'garbage':1, 'barrier':2, 'damaged_footpath':3,
    'construction_waste':4, 'abandoned_vehicle':5, 'uneven_road':6, 'other':7
}

def convert_to_yolo(row):
    fname = row['filename']
    img = cv2.imread(str(IMG_DIR/fname))
    if img is None:
        return None
    h,w = img.shape[:2]
    xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
    x_center = (xmin + xmax)/2.0 / w
    y_center = (ymin + ymax)/2.0 / h
    bw = (xmax - xmin)/w
    bh = (ymax - ymin)/h
    cls = CLASS_MAP[row['class']]
    return f"{cls} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}", img, xmin,ymin,xmax,ymax,cls

def main(csv_path='data/train.csv'):
    df = pd.read_csv(csv_path)
    for fname, group in df.groupby('filename'):
        txt_path = OUT_LABELS / (Path(fname).stem + '.txt')
        lines = []
        for _, r in group.iterrows():
            res = convert_to_yolo(r)
            if res is None:
                continue
            line, img, xmin,ymin,xmax,ymax,cls = res
            lines.append(line)
            crop = img[ymin:ymax, xmin:xmax]
            outfn = CROPS_DIR / f"{Path(fname).stem}_{_}_{cls}.jpg"
            cv2.imwrite(str(outfn), crop)
        if lines:
            txt_path.write_text("\n".join(lines))

if __name__ == "__main__":
    main()
