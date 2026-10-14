import numpy as np
import cv2 

X_RANGE = (-3.0, 3.0)              # metri di strada: (sinistra, destra) rispetto all'auto
Z_RANGE = (7.0, 30.0)              # metri in avanti: (vicino, lontano). Sotto i ~7 m il                                   # bordo basso dell'immagine non vede piu' la strada
PX_PER_M = 30                      # risoluzione della bird-eye view (1 px = 1/30 m)
GAUSS_KSIZE = 5                    # lato del nucleo gaussiano (dispari)
SOGLIA_K = 2.0                     # soglia = media(G) + K * deviazione standard(G)

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

def brid_eye_view(img, trapezio, x_range=X_RANGE, z_range=Z_RANGE, px_per_m=PX_PER_M):
    """Trasforma la prospettiva dell'immagine in una vista dall'alto (bird-eye view)."""
    # dimensioni della bird-eye view in pixel
    width = int((x_range[1] - x_range[0]) * px_per_m)
    height = int((z_range[1] - z_range[0]) * px_per_m)
    trapezio_dst = np.float32([[0, height], [width, height], [width, 0], [0, 0]])  # rettangolo in alto
    M = cv2.getPerspectiveTransform(trapezio, trapezio_dst)      # matrice di trasformazione
    M_inv = cv2.getPerspectiveTransform(trapezio_dst, trapezio)  # matrice inversa
    bew =  cv2.warpPerspective(img, M, (width, height))
    return bew, M, M_inv

#segmentation 
def gaussian_filter(img, ksize=GAUSS_KSIZE):
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.GaussianBlur(grey, (ksize, ksize), 0)

def gradient_filter(img):
    """Calcola il gradiente orizzontale e verticale, e restituisce la magnitudine."""
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # gradiente orizzontale
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # gradiente verticale
    return np.sqrt(gx**2 + gy**2)               # magnitudine del gradiente

def treshold(img, k=SOGLIA_K):
    """Soglia: media + k * deviazione standard."""
    mean = np.mean(img)
    std = np.std(img)
    treshold = mean + k * std
    return (img > treshold).astype(np.uint8) * 255