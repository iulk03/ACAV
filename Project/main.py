import matplotlib
matplotlib.use("Agg")  # file-only backend: no popup windows

import glob
from pathlib import Path
import cv2
import numpy as np
import lane_fitting as lf
import lane_pipeline as lp
import obstacle_detection as od


BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR / "data_road"
OUT_DIR = BASE_DIR / "debug_images"
SET = "training"     # "training" oppure "testing"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OPTIONS = (
    {"um": 95, "umm": 96, "uu": 98}
    if SET == "training"
    else {"um": 96, "umm": 94, "uu": 100}
)

# delete previously generated images
for file_path in OUT_DIR.iterdir():
    if file_path.is_file():
        file_path.unlink()
if OUT_DIR.is_dir() and next(OUT_DIR.iterdir(), None) is None:
    print("Folder emptied")

# number --> one image
# None   --> all the images of the specificied type
# 'w'    --> only wrongly detected images
SINGLE = 'w'
TYPE = 0        # 0 = um, 1 = umm, 2 = uu
ALL = True      # True to process all the images (um, umm, uu)
DEBUG = not ALL and SINGLE is not None   # full debug images only when few images are processed
wrong = {
    "um": [12, 43, 44, 45],
    "umm": [13, 45],
    "uu": [6, 7, 8, 17, 27, 29, 59, 69, 71, 92, 93, 94, 97],
}

marks = {}          # type -> [(img_idx, interior marking support)]
supports = {}       # type -> [(img_idx, max candidate support)]

# Inizializza detector COCO (torchvision) una sola volta
detector = od.ObstacleDetector(conf=0.5)


def ego_width_stats():
    """Ego-mask width (m) near the car (z=7) vs far (z=20) on every image. No plots, no road mask, no fitting."""
    for img_type, n in OPTIONS.items():
        files = sorted(glob.glob(f"{ROOT}/{SET}/image_2/{img_type}_*.png"))
        res = []
        for idx in range(n):
            f = Path(files[idx])
            img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            trapezoid = lp.compute_trapezoid(lp.read_calib(ROOT / SET / "calib" / f"{f.stem}.txt"))
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


def report():
    for t, res in marks.items():
        m = np.array([r[1] for r in res if r[1] >= 0])
        if len(m):
            print(f"[{t}] interior marking support: min {m.min():.0f}, p25 {np.percentile(m, 25):.0f}, median {np.median(m):.0f}, p75 {np.percentile(m, 75):.0f}, max {m.max():.0f}; "
                  f"{(m < lf.MARKING_MIN_SUPPORT).sum()}/{len(m)} under the gate ({lf.MARKING_MIN_SUPPORT:.0f})")
    print()
    for t, res in supports.items():
        m = np.array([r[1] for r in res])
        print(f"[{t}] max candidate support: min {m.min():.0f}, p25 {np.percentile(m, 25):.0f}, median {np.median(m):.0f}, p75 {np.percentile(m, 75):.0f}, max {m.max():.0f}")
        print(f"[{t}] lowest 10: " + ", ".join(f"{i} ({s:.0f})" for i, s in sorted(res, key=lambda r: r[1])[:10]))


def execute(img_idx, img_type):
    files = sorted(glob.glob(f"{ROOT}/{SET}/image_2/{img_type}_*.png"))
    image_path = Path(files[img_idx])
    stem = image_path.stem
    img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    calib_path = ROOT / SET / "calib" / f"{stem}.txt"

    # ---- 1) Obstacle detection eseguita PRIMA della lane detection ----
    raw_dets = detector.detect(img)

    # ---- 2) Pipeline BEV + fitting corsia guidati anche dagli ostacoli ----
    M, M_inv, thresh, guide = lp.lane_pipeline(
        calib_path,
        img,
        out_path=OUT_DIR / f"{stem}_bev.png" if DEBUG else None,
        raw_dets=raw_dets,
    )
    out = lf.lane_fitting(thresh, img, M_inv, guide)

    # ---- 3) Valutazione ostacoli rispetto alla corsia predetta ----
    obstacles = od.evaluate_obstacles(raw_dets, M, curved=out.get("curved"))
    out["obstacles"] = obstacles
    in_lane_obs = [o for o in obstacles if o["in_ego_lane"]]

    if DEBUG:
        lf.plot_bev_debug(thresh, out, OUT_DIR / f"{stem}_candidates.png")
        lf.plot_candidates_thresh(thresh, out, OUT_DIR / f"{stem}_cands_thresh.png")
        lf.plot_candidates_image(img, out, M_inv, OUT_DIR / f"{stem}_cands_img.png")

    iou_txt = ""
    if out.get("road_frac") is not None:
        iou_txt += f", road_frac: {out['road_frac']:.2f}" + (f" [ALT: {out['alt']}]" if out.get("alt") else "")
    sup = out.get("supports", [])
    supports.setdefault(img_type, []).append((img_idx, sup[0] if sup else 0.0))
    ms = out.get("marking_support")
    marks.setdefault(img_type, []).append((img_idx, -1.0 if ms is None else ms))
    iou_txt += f", marks: {'n/a' if ms is None else f'{ms:.0f}'}" + (" [NO MARKS]" if out.get("no_marks") else "")
    src = (out.get("guide") or {}).get("source")
    iou_txt += f", guide: {src}, supp: [{', '.join(f'{s:.0f}' for s in sup[:3])}]"

    obs_txt = f", obs: {len(obstacles)}"
    if in_lane_obs:
        closest = in_lane_obs[0]
        obs_txt += f" [IN LANE: {closest['cls']} @ {closest['distance_m']:.1f}m]"

    base_vis = out["overlay"] if "overlay" in out else img
    final_vis = od.draw_obstacles(base_vis, obstacles)

    if "overlay" in out:
        cv2.imwrite(str(OUT_DIR / f"{stem}_overlay.png"), cv2.cvtColor(final_vis, cv2.COLOR_RGB2BGR))
        print(f"type: {img_type}, img: {img_idx}, ego found, estimated side: {out['stimata']}{iou_txt}{obs_txt}")
    else:
        if obstacles:
            cv2.imwrite(str(OUT_DIR / f"{stem}_overlay.png"), cv2.cvtColor(final_vis, cv2.COLOR_RGB2BGR))
        print(f"type: {img_type}, img: {img_idx}, no lane found. Points: {len(out['pts'])}; candidates: {len(out['candidates'])}{iou_txt}{obs_txt}")


if ALL:
    for type, max_img_idx in list(OPTIONS.items()):
        if SINGLE is None:
            for idx in range(max_img_idx):
                execute(idx, type)
        elif SINGLE == "w":
            for idx in wrong[type]:
                execute(idx, type)
        else:
            execute(SINGLE, type)
else:
    type, max_img_idx = list(OPTIONS.items())[TYPE]
    if SINGLE is None:
        for idx in range(max_img_idx):
            execute(idx, type)
    elif SINGLE == "w":
        for idx in wrong[type]:
            execute(idx, type)
    else:
        execute(SINGLE, type)


report()