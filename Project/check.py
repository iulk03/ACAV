from pathlib import Path
import cv2
import lane_fitting
import lane_pipeline as lp

BASE_DIR = Path(__file__).resolve().parent
files = sorted((BASE_DIR / "data_road/training/image_2").glob("*.png"))
zs = []
for f in files:
    img = cv2.imread(str(f))
    calib = lp.read_calib(BASE_DIR / "data_road/training/calib" / f"{f.stem}.txt")
    zs.append(lp.z_at_image_bottom(calib, img.shape[0]))
print(f"min {min(zs):.2f}  max {max(zs):.2f}  mean {sum(zs)/len(zs):.2f}")
print(sum(z > 6.5 for z in zs), "su", len(zs))

BASE_DIR = Path(__file__).resolve().parent
wrong = [10, 11, 12, 28, 29, 30, 33, 34, 38, 40, 45, 47, 61, 62, 64, 65, 76, 93]

files = sorted((BASE_DIR / "data_road/training/image_2").glob("um_*.png"))
for i in wrong:
    f = files[i]
    img = cv2.imread(str(f))
    calib = lp.read_calib(BASE_DIR / "data_road/training/calib" / f"{f.stem}.txt")
    z = lp.z_at_image_bottom(calib, img.shape[0])
    print(f"{f.stem}: z_bottom = {z:.2f}" + ("   <-- sopra 6.5" if z > 6.5 else ""))