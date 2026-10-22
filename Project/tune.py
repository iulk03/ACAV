import glob, random
from pathlib import Path
import cv2, numpy as np
import lane_pipeline as lp
import lane_fitting as lf

ROOT = Path(__file__).resolve().parent / "data_road" / "training"
files = sorted(glob.glob(str(ROOT / "image_2" / "um_*.png")))
train, val = files[0::2], files[1::2]          # metà e metà

def gt_lane_mask(img_path):
    stem = Path(img_path).stem.replace("um_", "um_lane_")
    gt = cv2.imread(str(ROOT / "gt_image_2" / f"{stem}.png"))
    return gt[..., 0] > 0     # BEWARE: controlla visivamente che sia la corsia

def pred_mask(out, M_inv, shape):
    m = np.zeros(shape, np.uint8)
    if out.get("ransac_sx") is None or out.get("ransac_dx") is None:
        return m.astype(bool)
    lf_pts = lf.line_on_image(out["ransac_sx"], M_inv)
    rg_pts = lf.line_on_image(out["ransac_dx"], M_inv)
    area = np.vstack((lf_pts, rg_pts[::-1])).astype(np.int32)
    cv2.fillPoly(m, [area], 1)
    return m.astype(bool)

def iou_one(path, p):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    calib = ROOT / "calib" / f"{Path(path).stem}.txt"
    M_inv, th, grey, trap = lp.lane_pipeline(calib, img, show=False,
                                             clip=p["clip"], percentile=p["perc"])
    for k in ("W_CENTER", "SLOPE_DELTA_MAX", "N_CANDIDATES","EGO_TOL"):
        setattr(lf, k, p[k])
    out = lf.lane_fitting(th, img, M_inv)
    pred = pred_mask(out, M_inv, img.shape[:2])
    gt = gt_lane_mask(path)
    roi = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(roi, [trap.astype(np.int32)], 1)      # valuta solo dentro il trapezio
    roi = roi.astype(bool)
    pred, gt = pred & roi, gt & roi
    union = (pred | gt).sum()
    return (pred & gt).sum() / union if union else 0.0

def score(p, subset):
    return float(np.mean([iou_one(f, p) for f in subset]))

def sample():
    return dict(clip=random.choice([1.0, 1.5, 2.0, 3.0]),
                perc=random.choice([97, 98, 98.5, 99]),
                W_CENTER=random.choice([10, 20, 40, 100]),
                SLOPE_DELTA_MAX=random.choice([0.06, 0.1, 0.15]),
                N_CANDIDATES=random.choice([5, 8]),
                EGO_TOL=random.choice([0.3, 0.8, 1.5, 3.0, 100.0]))

random.seed(0)
best, best_s = None, -1
for t in range(60):
    p = sample()
    s = score(p, train)
    print(t, round(s, 3), p)
    if s > best_s:
        best, best_s = p, s

print("MIGLIORI:", best, "train IoU:", round(best_s, 3))
print("validation IoU:", round(score(best, val), 3))

scores = sorted((iou_one(f, best), f) for f in val)
print("PEGGIORI 8:")
for s, f in scores[:8]:
    print(round(s, 3), Path(f).name)
print("mediana:", round(float(np.median([s for s, _ in scores])), 3))
print("immagini con IoU < 0.1:", sum(s < 0.1 for s, _ in scores), "su", len(scores))