import glob
from pathlib import Path
import cv2
import lane_fitting as lf
import lane_pipeline as lp

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR / "data_road"
OUT_DIR = BASE_DIR / "debug_images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OPTIONS = {'um':94, 'uum':95, 'uu':97}

# set this variable to a number to process one image, None to process all the images, w to process only wrongly detected images
SINGLE = 'w'
TYPE, TOT_IMG = list(OPTIONS.items())[0] # depending which type of raod we want to check

type = sorted(glob.glob(f"{ROOT}/training/image_2/{TYPE}_*.png"))

wrong = [44, 70]

def execute(img_idx, name_flag):
    image_path = Path(type[img_idx])
    stem = image_path.stem
    img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    calib_path = ROOT / "training" / "calib" / f"{stem}.txt"

    pipe = lp.lane_pipeline(calib_path, img, out_path=OUT_DIR / f"{stem}_bev.png")
    out = lf.lane_fitting(pipe, img)

    lf.plot_bev_debug(pipe["thresh"], out, OUT_DIR / f"{stem}_candidates.png")
    lf.save_candidates(img, pipe["thresh"], out["pts"], out["candidates"] + out["edges"], pipe["M_inv"],
                        OUT_DIR / f"00_{stem}_cands_bev.png", OUT_DIR / f"{stem}_cands_img.png")

    if "overlay" in out:
        if name_flag: # I find this way easier to visualize the results in the folder
            cv2.imwrite(str(OUT_DIR / f"00_{stem}_overlay.png"), cv2.cvtColor(out["overlay"], cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(str(OUT_DIR / f"{stem}_overlay.png"), cv2.cvtColor(out["overlay"], cv2.COLOR_RGB2BGR))
        print(f"img: {img_idx}, lanes: {len(out['lanes'])} (ego found: {out['ransac_sx'] is not None})")
    else:
        print(f"No lane found. Points: {len(out['pts'])}; painted candidates: {len(out['candidates'])}; "f"road edges: {len(out['edges'])}")

if SINGLE is None:
    for idx in range(TOT_IMG):
        execute(idx, True)
elif SINGLE == 'w':
    for idx in wrong:
        execute(idx, True)
else:
    execute(SINGLE, False)