import matplotlib
matplotlib.use("Agg")  # file-only backend: no popup windows

import glob
from pathlib import Path
import cv2
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
SINGLE = 87
TYPE = 0 # 0 = um, 1 = umm, 2 = uu
ALL = False # True to process all the images (um, umm, uu)

wrong = {'um': [49, 76],
        'umm': [8, 12, 22, 23, 24, 25, 42, 52, 71, 82, 90],
        'uu' : [1, 3, 5, 6, 7, 8, 9, 10, 13, 15, 16, 17, 23, 27, 28, 29, 30, 36, 37, 42, 49, 51, 54, 57, 83, 84, 85, 86, 87, 88, 89, 90, 93, 94, 96]}

def execute(img_idx, img_type):
    files = sorted(glob.glob(f"{ROOT}/training/image_2/{img_type}_*.png"))
    image_path = Path(files[img_idx])
    stem = image_path.stem
    img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    calib_path = ROOT / "training" / "calib" / f"{stem}.txt"

    M_inv, thresh, guide = lp.lane_pipeline(calib_path, img, out_path=OUT_DIR / f"{stem}_bev.png")
    out = lf.lane_fitting(thresh, img, M_inv, guide)
    lf.plot_bev_debug(thresh, out, OUT_DIR / f"{stem}_candidates.png")
    lf.plot_candidates_thresh(thresh, out, OUT_DIR / f"{stem}_cands_thresh.png")
    lf.plot_candidates_image(img, out, M_inv, OUT_DIR / f"{stem}_cands_img.png")

    if "overlay" in out:
        cv2.imwrite(str(OUT_DIR / f"{stem}_overlay.png"), cv2.cvtColor(out["overlay"], cv2.COLOR_RGB2BGR))
        print(f"type: {img_type}, img: {img_idx}, ego found, estimated side: {out['stimata']}")
    else:
        print(f"type: {img_type}, img: {img_idx}, no lane found. Points: {len(out['pts'])}; candidates: {len(out['candidates'])}")


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


# TODO: check slope tra due rette 
#       eliminare linee nel canny se non sono linee rette