import glob
from pathlib import Path
import cv2
import lane_fitting as lf
import lane_pipeline as lp

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR / "data_road"
OUT_DIR = BASE_DIR / "debug_images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SINGLE = 6 # set this variable to a number to process one image, None to process all the images

um = sorted(glob.glob(f"{ROOT}/training/image_2/um_*.png"))

# wrong in: 6, 8, 9, 10, 14, 15, 17, 19(?), 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 37, 40, 41, 42, 43, 44, 45, 51, 55
#           56, 70, 71, 73, 75, 76, 77, 84, 85, 86, 87, 89, 90, 91, 92

def execute(img_idx):
    image_path = Path(um[img_idx])
    stem = image_path.stem
    img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    calib_path = ROOT / "training" / "calib" / f"{stem}.txt"

    pipe = lp.lane_pipeline(calib_path, img, out_path=OUT_DIR / f"{stem}_bev.png")
    out = lf.lane_fitting(pipe, img)

    lf.plot_bev_debug(pipe["thresh"], out, OUT_DIR / f"{stem}_candidates.png")
    lf.save_candidates(img, pipe["thresh"], out["pts"], out["candidates"] + out["edges"], pipe["M_inv"],
                        OUT_DIR / f"{stem}_cands_bev.png", OUT_DIR / f"{stem}_cands_img.png")

    if "overlay" in out:
        cv2.imwrite(str(OUT_DIR / f"{stem}_overlay.png"), cv2.cvtColor(out["overlay"], cv2.COLOR_RGB2BGR))
        print(f"img: {img_idx}, lanes: {len(out['lanes'])} (ego found: {out['ransac_sx'] is not None})")
    else:
        print(f"No lane found. Points: {len(out['pts'])}; painted candidates: {len(out['candidates'])}; "f"road edges: {len(out['edges'])}")

if SINGLE is None:
    for idx in range(94):
        execute(idx)
else:
    execute(SINGLE)