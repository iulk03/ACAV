import matplotlib
matplotlib.use("Agg")  # file-only backend: no popup windows

import glob
from pathlib import Path
import cv2
import numpy as np
import net_classifier as nc
import lane_fitting as lf
import lane_pipeline as lp


BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR / "data_road"
OUT_DIR = BASE_DIR / "debug_images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OPTIONS = {'um': 95, 'umm': 96, 'uu': 98}

# delete previously generated images
for file_path in OUT_DIR.iterdir():
    if file_path.is_file():
        file_path.unlink()
if OUT_DIR.is_dir() and next(OUT_DIR.iterdir(), None) is None:
    print('Folder emptied')

# number --> one image
# None   --> all the images of the specificied type
# 'w'    --> only wrongly detected images
SINGLE = None
TYPE = 0 # 0 = um, 1 = umm, 2 = uu
ALL = False # True to process all the images (um, umm, uu)
DEBUG = not ALL and SINGLE is not None   # full debug images only when few images are processed

wrong = {'um': [49, 76],
        'umm': [16, 15],
        'uu' : [1, 3, 5, 6, 7, 8, 9, 10, 13, 15, 16, 17, 23, 27, 28, 29, 30, 36, 37, 42, 49, 51, 54, 57, 83, 84, 85, 86, 87, 88, 89, 90, 93, 94, 96]}

IOU_FAIL = 0.5      # um: below this ego-lane IoU an image is considered wrong
ROAD_FAIL = 0.8     # umm / uu: below this on-road fraction an image is considered wrong
scores = {}         # type -> [(img_idx, score)]


def ego_width_stats():
    """Ego-mask width (m) near the car (z=7) vs far (z=20) on every image. No plots, no road mask, no fitting."""
    for img_type, n in OPTIONS.items():
        files = sorted(glob.glob(f"{ROOT}/training/image_2/{img_type}_*.png"))
        res = []
        for idx in range(n):
            f = Path(files[idx])
            img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            trapezoid = lp.compute_trapezoid(lp.read_calib(ROOT / "training" / "calib" / f"{f.stem}.txt"))
            bev, M, _ = lp.brid_eye_view(lp.apply_roi(img, trapezoid), trapezoid)
            valid = lp.valid_bev_mask(img.shape, M, (bev.shape[1], bev.shape[0]))
            ego = lp.ego_mask_bev(img, M, valid) > 0
            if not ego.any():
                print(f"{img_type} {idx}: ego mask empty")
                continue
            w = ego.sum(1) / lp.PX_PER_M                       # row 0 = z 20 m, last row = z 7 m
            res.append((idx, w[-10:].mean(), w[:10].mean()))
            print(f"{img_type} {idx}: near {res[-1][1]:.1f} m, far {res[-1][2]:.1f} m")
        d = np.array([r[1] - r[2] for r in res])
        print(f"[{img_type}] near-far: mean {d.mean():+.2f} m, median {np.median(d):+.2f} m, {(d > 1.0).sum()}/{len(d)} with near wider than far by >1 m")
        print(f"[{img_type}] narrowing = {[r[0] for r in res if r[1] - r[2] > 1.0]}")


def roi_mask(img, calib_path, valid):
    """Pixels of the original image inside the BEV trapezoid (z 7-20 m), where the lane is predicted, and inside the GT valid area."""
    roi = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(roi, [lp.compute_trapezoid(lp.read_calib(calib_path)).astype(np.int32)], 1)
    return (roi > 0) & valid


def image_score(image_path, img, out, calib_path, img_type):
    """Metric that needs no manual check.
    um      : IoU between the predicted lane polygon and the KITTI ego-lane GT (restricted to the ROI).
    umm, uu : there is no ego-lane GT, only the road GT. Score = fraction of the predicted polygon that lies ON THE ROAD.
              It catches a lane drawn outside the road, not the wrong lane of a multi-lane road (umm): for those the
              score stays high, so look at the lowest ones and at a few random ones anyway.
    A missing lane scores 0."""
    pred = out.get("pred_mask")
    if img_type == "um":
        lane, valid = nc.lane_gt(image_path)
        roi = roi_mask(img, calib_path, valid)
        if pred is None:
            return 0.0
        return float((pred & lane & roi).sum() / max(((pred | lane) & roi).sum(), 1))
    road, valid = nc.road_gt(image_path)
    roi = roi_mask(img, calib_path, valid)
    if pred is None:
        return 0.0
    return float((pred & road & roi).sum() / max((pred & roi).sum(), 1))


def report():
    for t, res in scores.items():
        label, fail = ("ego-lane IoU", IOU_FAIL) if t == "um" else ("on-road fraction", ROAD_FAIL)
        v = np.array([r[1] for r in res])
        bad = [i for i, r in res if r < fail]
        worst = sorted(res, key=lambda r: r[1])[:8]
        print(f"[{t}] {label}: mean {v.mean():.3f}, median {np.median(v):.3f}, {len(bad)}/{len(v)} under {fail}")
        print(f"[{t}] wrong = {bad}")
        print(f"[{t}] lowest 8: " + ", ".join(f"{i} ({s:.2f})" for i, s in worst))


def execute(img_idx, img_type):
    files = sorted(glob.glob(f"{ROOT}/training/image_2/{img_type}_*.png"))
    image_path = Path(files[img_idx])
    stem = image_path.stem
    img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    calib_path = ROOT / "training" / "calib" / f"{stem}.txt"

    M_inv, thresh, guide = lp.lane_pipeline(calib_path, img, out_path=OUT_DIR / f"{stem}_bev.png" if DEBUG else None)
    out = lf.lane_fitting(thresh, img, M_inv, guide)
    if DEBUG:
        lf.plot_bev_debug(thresh, out, OUT_DIR / f"{stem}_candidates.png")
        lf.plot_candidates_thresh(thresh, out, OUT_DIR / f"{stem}_cands_thresh.png")
        lf.plot_candidates_image(img, out, M_inv, OUT_DIR / f"{stem}_cands_img.png")

    sc = image_score(image_path, img, out, calib_path, img_type)
    iou_txt = f", {'IoU' if img_type == 'um' else 'on-road'}: {sc:.2f}"
    scores.setdefault(img_type, []).append((img_idx, sc))

    if "overlay" in out:
        cv2.imwrite(str(OUT_DIR / f"{stem}_overlay.png"), cv2.cvtColor(out["overlay"], cv2.COLOR_RGB2BGR))
        print(f"type: {img_type}, img: {img_idx}, ego found, estimated side: {out['stimata']}{iou_txt}")
    else:
        print(f"type: {img_type}, img: {img_idx}, no lane found. Points: {len(out['pts'])}; candidates: {len(out['candidates'])}{iou_txt}")


if ALL:
    for type, max_img_idx in list(OPTIONS.items()):
        for idx in range(max_img_idx):
            execute(idx, type)
else:
    type, max_img_idx = list(OPTIONS.items())[TYPE] # depending which type of road we want to check
    if SINGLE is None:
        for idx in range(max_img_idx):
            execute(idx, type)
    elif SINGLE == 'w':
        for idx in wrong[type]:
            execute(idx, type)
    else:
        execute(SINGLE, type)


report()


# TODO: check slope tra due rette 
#       eliminare linee nel canny se non sono linee rette