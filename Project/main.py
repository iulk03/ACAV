import matplotlib.pyplot as plt
import cv2 
import glob
from pathlib import Path
import lane_pipeline as lp
import lane_fitting as lf

BASE_DIR = Path(__file__).resolve().parent   
ROOT = BASE_DIR / "data_road"
OUT_DIR = BASE_DIR / "debug_images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

um = sorted(glob.glob(f"{ROOT}/training/image_2/um_*.png"))
umm = sorted(glob.glob(f"{ROOT}/training/image_2/umm_*.png"))
uu = sorted(glob.glob(f"{ROOT}/training/image_2/uu_*.png"))

IMAGE_INDEX = 8

image_path = Path(um[IMAGE_INDEX])
img = cv2.imread(str(image_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

calib_path = ROOT / "training" / "calib" / f"{image_path.stem}.txt"
M_inv, img_thresh, img_gray = lp.lane_pipeline(calib_path, img, out_path=OUT_DIR / f'{image_path.stem}_bev.png')

out = lf.lane_fitting(img_thresh, img, M_inv, img_gray)
lf.plot_bev_debug(img_thresh, out, OUT_DIR / f'{image_path.stem}_candidates.png')

if "overlay" in out:
    '''plt.figure(figsize=(12, 4))
    plt.imshow(out["overlay"])
    plt.axis("off")
    plt.show()'''
    cv2.imwrite(
        str(OUT_DIR / f"{image_path.stem}_overlay.png"),
        cv2.cvtColor(out["overlay"], cv2.COLOR_RGB2BGR),
    )
else:
    print(
    "RANSAC non ha trovato entrambe le linee. "
    f"Punti totali: {len(out['pts'])}; " # array of all points
    f"candidati trovati: {len(out['candidates'])}; " # line candidates from ransac algo
    f"linea sinistra trovata: {out['ransac_sx'] is not None}; "
    f"linea destra trovata: {out['ransac_dx'] is not None}"
)

lf.save_candidates(
    img, img_thresh, out["pts"], out["candidates"], M_inv,
    OUT_DIR / f"{image_path.stem}_cands_bev.png",
    str(OUT_DIR / f"{image_path.stem}_cands_img.png"),
)