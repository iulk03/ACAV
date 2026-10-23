import glob
from pathlib import Path
import cv2
import lane_fitting as lf
import lane_pipeline as lp

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR / "data_road"
OUT_DIR = BASE_DIR / "debug_images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SINGLE = 3 # set this variable to a number to process one image, None to process all the images

um = sorted(glob.glob(f"{ROOT}/training/image_2/um_*.png"))

# wrong in: 0, 1, 2, 3, 5, 7, 11, 12, 14, 16, 17, 18, 19, 42, 47, 52, 53, 72, 75, 76,
#           80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91, 92

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