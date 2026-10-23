from pathlib import Path
import glob
import cv2
import numpy as np
import lane_pipeline as lp
import lane_fitting as lf

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR / "data_road"
TRAIN = ROOT / "training"


def score(pred, gt):
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


def evaluate_image(image_path):
    img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    stem = image_path.stem                                   # e.g. um_000000
    calib_path = TRAIN / "calib" / f"{stem}.txt"
    gt_path = TRAIN / "gt_image_2" / f"{stem.replace('_', '_lane_', 1)}.png"
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) > 0

    M_inv, img_thresh, img_gray = lp.lane_pipeline(calib_path, img)
    out = lf.lane_fitting(img_thresh, img, M_inv, img_gray)

    if out["ransac_sx"] is not None and out["ransac_dx"] is not None:
        pred = lf.lane_mask(img.shape, out["ransac_sx"], out["ransac_dx"], M_inv)
    else:
        pred = np.zeros(img.shape[:2], dtype=bool)           # failure counts as F1 = 0
    return score(pred, gt)


def evaluate_subset(prefix, n_tuning=30):
    paths = sorted(Path(p) for p in glob.glob(str(TRAIN / "image_2" / f"{prefix}_*.png")))
    tuning = paths[:n_tuning]
    results = {p.stem: evaluate_image(p) for p in tuning}
    return results


if __name__ == "__main__":
    for prefix in ("um", "umm", "uu"):
        res = evaluate_subset(prefix)
        mean = np.mean(list(res.values()))
        worst = sorted(res.items(), key=lambda kv: kv[1])[:10]
        print(f"{prefix}: mean F1 = {mean:.3f} ({len(res)} images)")
        for name, f1 in worst:
            print(f"   {name}  F1={f1:.3f}")