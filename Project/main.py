import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import cv2 
import glob
from pathlib import Path
import lane_pipeline as lp
import lane_fitting as lf
import numpy as np

BASE_DIR = Path(__file__).resolve().parent   # cartella dello script
ROOT = BASE_DIR / "data_road"
um = sorted(glob.glob(f"{ROOT}/training/image_2/um_*.png"))
umm = sorted(glob.glob(f"{ROOT}/training/image_2/umm_*.png"))

IMAGE_INDEX = 34
image_path = Path(um[IMAGE_INDEX])
img = cv2.imread(str(image_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

calib_path = ROOT / "training" / "calib" / f"{image_path.stem}.txt"
M_inv, img_thresh = lp.lane_pipeline(calib_path, img)

outt = lf.lane_fitting(img_thresh, img, M_inv)

""""
if "overlay" in outt:
    plt.figure(figsize=(12, 4))
    plt.imshow(outt["overlay"])
    plt.axis("off")
    plt.show()
else:
    print("RANSAC non ha trovato entrambe le linee, nessun overlay")
"""

