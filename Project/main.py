import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import cv2 
import glob
from pathlib import Path
import lane_pipeline as lp

BASE_DIR = Path(__file__).resolve().parent   # cartella dello script
ROOT = BASE_DIR / "data_road"
um = sorted(glob.glob(f"{ROOT}/training/image_2/um_*.png"))
umm = sorted(glob.glob(f"{ROOT}/training/image_2/umm_*.png"))

IMAGE_INDEX = 34
image_path = Path(umm[IMAGE_INDEX])
img = cv2.imread(str(image_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

calib_path = ROOT / "training" / "calib" / f"{image_path.stem}.txt"
calib = lp.leggi_calib(calib_path)
trapezio = lp.calcola_trapezio(calib)
img_roi = lp.applica_roi(img, trapezio)
img_bird_eye, M, M_inv = lp.brid_eye_view(img_roi, trapezio)
img_filtered = lp.gaussian_filter(img_bird_eye)
img_grad = lp.gradient_filter(img_filtered)
img_thresh = lp.treshold(img_grad)

# Visualizzo
vis = img.copy()
cv2.polylines(vis, [trapezio.astype("int32")], True, (255, 0, 0), 3)
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].imshow(vis)
ax[0].set_title("ROI")
ax[1].imshow(img_grad)
ax[1].set_title("Dopo la BEW")
for a in ax:
    a.axis("off")
plt.show()
