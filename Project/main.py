import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2 
import os 
import glob
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent   # cartella dello script
ROOT = BASE_DIR / "data_road"
X_RANGE = (-3.0, 3.0)              # metri di strada: (sinistra, destra) rispetto all'auto
Z_RANGE = (7.0, 30.0)              # metri in avanti: (vicino, lontano). Sotto i ~7 m il
                                   # bordo basso dell'immagine non vede piu' la strada
PX_PER_M = 30                      # risoluzione della bird-eye view (1 px = 1/30 m)
GAUSS_KSIZE = 5                    # lato del nucleo gaussiano (dispari)
SOGLIA_K = 2.0                     # soglia = media(G) + K * deviazione standard(G)
OUT_DIR = "output_passi"           # dove salvare le immagini di ogni passo


# 1. Scarica e carica il dataset KITTI (split di addestramento)
# with_info=True restituisce anche i metadati strutturali del dataset
root = ROOT          # cartella estratta
print(os.listdir(root))                          # atteso: training, testing
print(os.listdir(f"{root}/training"))            # atteso: calib, gt_image_2, image_2

um = sorted(glob.glob(f"{root}/training/image_2/um_*.png"))
umm = sorted(glob.glob(f"{root}/training/image_2/umm_*.png"))
print(len(um), "UM,", len(umm), "UMM")
#dataset, info = tfds.load("kitti", split="train", with_info=True)

# 2. Estraiamo un singolo elemento per vedere la struttura dati
"""
for idx in range(5):
    img = cv2.imread(um[idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 4))
    plt.imshow(img)
    plt.title(Path(um[idx]).name)
    plt.axis("off")
    plt.show()
"""
def leggi_calib(path):
    """Legge le righe 'nome: numeri...' del file di calibrazione in un dizionario."""
    calib = {}
    with open(path) as f:
        for riga in f:
            if ":" in riga:
                nome, valori = riga.split(":", 1)
                calib[nome.strip()] = np.array(valori.split(), dtype=float)
    return calib

#roi selection step 1
def calcola_trapezio(calib, x_range=X_RANGE, z_range=Z_RANGE):
    """
    Decide DOVE guardare: un rettangolo reale sulla strada (x_range metri di
    larghezza, z_range metri di distanza), proiettato nell'immagine.
    Nei file di KITTI Road ci sono i dati della camera (P2) e la sua posizione
    rispetto alla strada (Tr_cam_to_road), cioe' le informazioni su campo visivo,
    posizione e orientazione di cui parlano gli appunti per la ROI.
    Restituisce i 4 vertici (u, v) in pixel: basso-sx, basso-dx, alto-dx, alto-sx.
    """
    P2 = calib["P2"].reshape(3, 4)                 # proiezione camera 2 (f, cx, cy, ...)
    T = calib["Tr_cam_to_road"].reshape(3, 4)      # camera -> strada: [R | t]
    R, t = T[:, :3], T[:, 3]
 
    def proietta(x, z):
        """Punto SULLA STRADA (x di lato, z in avanti, quota 0) -> pixel (u, v)."""
        p_strada = np.array([x, 0.0, z])           # z_world = 0, come negli appunti
        p_cam = R.T @ (p_strada - t)               # dal sistema strada al sistema camera
        u, v, w = P2 @ np.append(p_cam, 1.0)       # s * (u, v, 1) = P2 * (punto, 1)
        return [u / w, v / w]                      # dividere per w elimina il fattore s
 
    return np.float32([
        proietta(x_range[0], z_range[0]),          # vicino, sinistra  (in basso)
        proietta(x_range[1], z_range[0]),          # vicino, destra    (in basso)
        proietta(x_range[1], z_range[1]),          # lontano, destra   (in alto)
        proietta(x_range[0], z_range[1]),          # lontano, sinistra (in alto)
    ])
 
 
def applica_roi(img, trapezio):
    """Scarta tutto cio' che sta fuori dal trapezio (cielo, edifici, bordi)."""
    maschera = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(maschera, [trapezio.astype(np.int32)], 255)   # dentro = bianco
    return cv2.bitwise_and(img, img, mask=maschera)            # fuori = nero


IMAGE_INDEX = 34
image_path = Path(um[IMAGE_INDEX])

img = cv2.imread(str(image_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

calib_path = root / "training" / "calib" / f"{image_path.stem}.txt"
calib = leggi_calib(calib_path)
trapezio = calcola_trapezio(calib)
img_roi = applica_roi(img, trapezio)

# Visualizzo
vis = img.copy()
cv2.polylines(vis, [trapezio.astype("int32")], True, (255, 0, 0), 3)
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].imshow(vis)
ax[0].set_title("Trapezio")
ax[1].imshow(img_roi)
ax[1].set_title("Dopo la ROI")
for a in ax:
    a.axis("off")
plt.show()
