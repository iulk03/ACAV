import matplotlib.pyplot as plt
import cv2 
import glob
from pathlib import Path
import lane_pipeline as lp
import lane_fitting as lf

# ---------------- Configurazione Percorsi ----------------
BASE_DIR = Path(__file__).resolve().parent   
ROOT = BASE_DIR / "data_road"

# Caricamento dei dataset KITTI Road
um = sorted(glob.glob(f"{ROOT}/training/image_2/um_*.png"))
umm = sorted(glob.glob(f"{ROOT}/training/image_2/umm_*.png"))
uu = sorted(glob.glob(f"{ROOT}/training/image_2/uu_*.png"))

# Selezione immagine di test
IMAGE_INDEX = 10
image_path = Path(um[IMAGE_INDEX])
img = cv2.imread(str(image_path))   
if img is None:
    print(f"Errore nel caricamento dell'immagine: {image_path}")
    exit(1)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
calib_path = ROOT / "training" / "calib" / f"{image_path.stem}.txt"

# 1. Pipeline di rilevamento contorni (Bird-Eye View + CLAHE + Canny)
M_inv, img_thresh = lp.lane_pipeline(calib_path, img)

# 2. Fitting delle linee di corsia tramite RANSAC
out = lf.lane_fitting(img_thresh, img, M_inv)

# 3. Debug visivo in prospettiva Bird-Eye (BEV)
lf.plot_bev_debug(img_thresh, out)

# 4. Visualizzazione Risultato Finale o Gestione Errori
if "overlay" in out and out["overlay"] is not None:
    plt.figure(figsize=(12, 4))
    plt.imshow(out["overlay"])
    plt.title(f"Corsia Rilevata su {image_path.name} {'(Lato Stimato)' if out.get('stimata') else '(Misurata)'}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
else:
    n_punti = len(out['pts']) if 'pts' in out else 0
    sx_trovata = out.get('ransac_sx') is not None
    dx_trovata = out.get('ransac_dx') is not None
    print(
        "RANSAC non è riuscito a stimare la corsia.\n"
        f"- Punti totali estratti da Canny: {n_punti}\n"
        f"- Linea sinistra individuata: {sx_trovata}\n"
        f"- Linea destra individuata: {dx_trovata}"
    )