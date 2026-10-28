import glob
from pathlib import Path
import cv2
import lane_fitting as lf
import lane_pipeline as lp
import vehicle_detector as vd


BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR / "data_road"
OUT_DIR = BASE_DIR / "debug_images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OPTIONS = {'um':94, 'umm':95, 'uu':97}

for file_path in OUT_DIR.iterdir():
        if file_path.is_file():
            file_path.unlink()
# set this variable to a number to process one image, None to process all the images for a specific type, w to process only wrongly detected images
SINGLE = 'w'
TYPE = 0 # 0 = um, 1 = umm, 2 = uu
ALL = False # True to process all the images (um, umm, uu)

wrong = [10, 11, 12, 28, 29, 30, 33, 34, 38, 40, 45, 47, 61, 62, 64, 65, 76, 93]

def execute(img_idx, img_type):
    type = sorted(glob.glob(f"{ROOT}/training/image_2/{img_type}_*.png"))
    image_path = Path(type[img_idx])
    stem = image_path.stem
    img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    calib_path = ROOT / "training" / "calib" / f"{stem}.txt"

    pipe = lp.lane_pipeline(calib_path, img, out_path=OUT_DIR / f"{stem}_bev.png")

    vehicles = lf.vehicles_to_road(vd.detect_vehicles(img), pipe["M"])
    out = lf.lane_fitting(pipe, img, vehicles)
    lf.print_candidate_stats(out)

    lf.plot_bev_debug(pipe["thresh"], out, OUT_DIR / f"{stem}_candidates.png")
    lf.save_candidates(img, pipe["thresh"], out["pts"], out["candidates"] + out["edges"], pipe["M_inv"], OUT_DIR / f"00_{stem}_cands_bev.png", OUT_DIR / f"{stem}_cands_img.png")

    if "overlay" in out:
        cv2.imwrite(str(OUT_DIR / f"{stem}_overlay.png"), cv2.cvtColor(out["overlay"], cv2.COLOR_RGB2BGR))
        print(f"img: {img_idx}, lanes: {len(out['lanes'])} (ego found: {out['ransac_sx'] is not None})")
    else:
        print(f"No lane found. Points: {len(out['pts'])}; painted candidates: {len(out['candidates'])}; "f"road edges: {len(out['edges'])}")


if ALL:
    for type, max_img_idx in list(OPTIONS.items()):
        for idx in range(max_img_idx):
            execute(idx, type)
else:
    type, max_img_idx = list(OPTIONS.items())[TYPE] # depending which type of raod we want to check
    if SINGLE is None:
        for idx in range(max_img_idx):
            execute(idx, type)
    elif SINGLE == 'w':
        for idx in wrong:
            execute(idx, type)
    else:
        execute(SINGLE, type)