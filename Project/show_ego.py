"""
Mostra la maschera ego sull'immagine originale.

Uso (dalla cartella del progetto, dove ci sono ego_net.pt e road_net.pt):
    python show_ego.py                 # TUTTE le immagini (um, umm, uu)
    python show_ego.py um 44 uu 10     # solo queste (coppie tipo / indice)
Salva debug_images/<stem>_ego.png

Pannelli:
  1. probabilita' grezza della rete ego (heatmap) sull'immagine
  2. grezza dopo soglia (nc.ROAD_THR) + trapezio ROI
  3. maschera ego FINALE (quella usata dalla pipeline, dopo pulizia nel BEV riportata sull'immagine, in azzurro)
     + road mask (verde) + GT corsia ego se esiste (contorno giallo, solo um)
"""
import sys
import glob
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import net_classifier as nc
import lane_pipeline as lp

ROOT = nc.ROOT
OUT_DIR = Path(__file__).resolve().parent / "debug_images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for file_path in OUT_DIR.iterdir():
    if file_path.is_file():
        file_path.unlink()
if OUT_DIR.is_dir() and next(OUT_DIR.iterdir(), None) is None:
    print('Folder emptied')

def blend(img, mask, color, alpha=0.45):
    out = img.copy()
    m = mask > 0
    out[m] = ((1 - alpha) * out[m] + alpha * np.array(color)).astype(np.uint8)
    return out


def show_ego(img_type, idx):
    files = sorted(glob.glob(f"{ROOT}/training/image_2/{img_type}_*.png"))
    path = Path(files[idx])
    stem = path.stem
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    calib = lp.read_calib(ROOT / "training" / "calib" / f"{stem}.txt")
    trap = lp.compute_trapezoid(calib)
    bev, M, M_inv = lp.brid_eye_view(lp.apply_roi(img, trap), trap)
    valid = lp.valid_bev_mask(img.shape, M, (bev.shape[1], bev.shape[0]))

    # 1-2) uscita grezza della rete, a risoluzione originale
    p_raw = nc.ego_prob_img(img)
    raw = (p_raw > nc.ROAD_THR).astype(np.uint8)
    heat = cv2.cvtColor(cv2.applyColorMap((p_raw * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    heat_ov = cv2.addWeighted(img, 0.55, heat, 0.45, 0)

    raw_ov = blend(img, raw, (0, 191, 255))
    cv2.polylines(raw_ov, [trap.astype(np.int32)], True, (255, 0, 0), 2)

    # 3) maschere finali (calcolate nel BEV) riportate nell'immagine
    ego_bev = lp.ego_mask_bev(img, M, valid)
    road_bev = nc.road_mask_bev(bev, valid)
    back = lambda m: cv2.warpPerspective(m, M_inv, (w, h), flags=cv2.INTER_NEAREST)
    final_ov = blend(img, back(road_bev), (0, 255, 0), alpha=0.30)
    final_ov = blend(final_ov, back(ego_bev), (0, 191, 255), alpha=0.50)
    if not ego_bev.any():
        print(f"{stem}: maschera ego FINALE vuota (la rete non trova nulla o non passa i controlli)")

    gt = nc.lane_gt(path)
    if gt is not None:
        cnts, _ = cv2.findContours(gt[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(final_ov, cnts, -1, (255, 255, 0), 2)

    panels = [("1. Ego prob (rete, grezza)", heat_ov),
              (f"2. Ego > {nc.ROAD_THR} + ROI", raw_ov),
              ("3. Ego finale (azzurro) + road (verde)" + (" + GT (giallo)" if gt is not None else ""), final_ov)]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for a, (title, im) in zip(axes, panels):
        a.imshow(im)
        a.set_title(title)
        a.axis("off")
    plt.tight_layout()
    out = OUT_DIR / f"{stem}_ego.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"salvato {out}")


def n_images(img_type):
    return len(glob.glob(f"{ROOT}/training/image_2/{img_type}_*.png"))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        jobs = [(t, int(i)) for t, i in zip(args[::2], args[1::2])]
    else:
        jobs = [(t, i) for t in ("um", "umm", "uu") for i in range(n_images(t))]
    for t, i in jobs:
        show_ego(t, i)
