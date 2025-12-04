
# src/preprocess.py
import cv2

def preprocess_image(img_bgr):
    """
    Simple preprocessing used during training:
    - 3x3 blur
    - CLAHE on L channel of LAB color space
    Returns a BGR image (same dtype as input).
    """
    if img_bgr is None:
        return None
    den = cv2.blur(img_bgr, (3,3))
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return out

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        img = cv2.imread(sys.argv[1])
        out = preprocess_image(img)
        cv2.imwrite(sys.argv[2], out)
