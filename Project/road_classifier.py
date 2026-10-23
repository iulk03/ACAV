"""
Classificatore pixel-per-pixel strada / non-strada (KITTI Road) e
estrazione dei punti candidati dal contorno della strada nella bird-eye view.

Il classificatore viene addestrato direttamente sulla BEV (immagine e GT warpate
con la stessa matrice M della pipeline), cosi' train e inferenza vedono lo stesso dominio.

Addestramento (una volta sola):
    python road_classifier.py
-> salva road_clf.joblib accanto a questo file.
"""
import glob
from pathlib import Path

import cv2
import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR / "data_road"
MODEL_PATH = BASE_DIR / "road_clf.joblib"

N_NEAR = 3000        # hard negatives: non-road pixels close to the road (sidewalks, bike lanes)
NEAR_PX = 30         # ~1 m
N_RANDOM = 2000      # pixel casuali per immagine
N_BOUNDARY = 2000    # pixel vicino al bordo strada (e' li' che si decide tutto)
BAND_PX = 12         # semi-larghezza della banda attorno al bordo, in pixel (BEV: 30 px = 1 m)
LOCAL_PX = 9         # finestra per le feature di texture locale
CONTEXT_PX = 31      # finestra piu' grande: contesto
ROAD_THR = 0.5       # soglia sulla probabilita' di strada
CHROMA_MIN = 40.0    # soglia sul gradiente di colore (Lab OpenCV, Sobel 3x3): alzala se entrano bordi d'ombra, abbassala se manca il bordo asfalto/erba
SEED = 0


# ---------------- feature e ground truth ----------------
def pixel_features(rgb):
    """24 feature per pixel: colore (RGB, HSV, Lab, log-cromaticita'), texture locale e di contesto, riga (z)."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)

    def local_std(win):
        k = (win, win)
        return np.sqrt(np.maximum(cv2.blur(L * L, k) - cv2.blur(L, k) ** 2, 0))

    # un'ombra scurisce i tre canali quasi nello stesso modo: i rapporti cambiano poco
    f = rgb.astype(np.float32) + 1.0
    chroma = np.dstack((np.log(f[:, :, 0] / f[:, :, 1]), np.log(f[:, :, 2] / f[:, :, 1])))

    # l'erba nella BEV diventa strisce verticali (molta energia in x, poca in y); l'asfalto no
    gx = np.abs(cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3))
    grad = np.dstack([cv2.blur(g, (win, win)) for g in (gx, gy) for win in (LOCAL_PX, CONTEXT_PX)])

    # la sfocatura dipende dalla distanza: la texture va letta in funzione di z (solo la riga, niente colonna)
    row = np.repeat(np.linspace(0, 1, h, dtype=np.float32), w).reshape(h, w, 1)

    return np.dstack((rgb, hsv, lab, cv2.blur(lab, (LOCAL_PX, LOCAL_PX)), local_std(LOCAL_PX)[..., None],
                      chroma, grad, local_std(CONTEXT_PX)[..., None],
                      cv2.blur(lab, (CONTEXT_PX, CONTEXT_PX)), row)).reshape(-1, 24).astype(np.float32)


def add_shadows(img, rng):
    """Ombre/chiazze di sole sintetiche (solo sull'immagine, la GT non cambia): data augmentation."""
    h, w = img.shape[:2]
    noise = 0.
    for cell in (20, 60):                                   # chiazze piccole + zone d'ombra grandi
        n = rng.random((h // cell + 2, w // cell + 2)).astype(np.float32)
        noise = noise + cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise / 2, (0, 0), 3)
    thr = np.quantile(noise, rng.uniform(0.3, 0.7))
    shade = np.clip((noise - thr) / rng.uniform(0.03, 0.08) * 0.5 + 0.5, 0, 1)[..., None]
    dark = rng.uniform(0.35, 0.7)                           # quanto scurisce l'ombra
    tint = np.array([0.92, 0.97, 1.05], np.float32)         # le ombre sono un po' piu' blu
    out = img.astype(np.float32) * (1 - dark * shade) * (1 + (tint - 1) * shade)
    out *= rng.uniform(0.85, 1.15)                          # esposizione globale
    return np.clip(out, 0, 255).astype(np.uint8)


def road_gt(image_path):
    """(road, valid) booleani dalla GT di KITTI Road. Nome: um_000093 -> um_road_000093."""
    p = Path(image_path)
    prefix, num = p.stem.split("_")
    gt_path = p.parents[1] / "gt_image_2" / f"{prefix}_road_{num}.png"
    gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
    # OpenCV legge BGR: canale 0 (blu) = strada, canale 2 (rosso) = area valida della GT
    return gt[:, :, 0] > 0, gt[:, :, 2] > 0
    

def sample_pixels(img, road, valid, rng):
    r8 = road.astype(np.uint8)
    k = np.ones((2 * BAND_PX + 1, 2 * BAND_PX + 1), np.uint8)
    band = (cv2.dilate(r8, k) != cv2.erode(r8, k)) & valid
    kn = np.ones((2 * NEAR_PX + 1, 2 * NEAR_PX + 1), np.uint8)
    near = (cv2.dilate(r8, kn) > 0) & ~road & valid
    pools = [(np.flatnonzero(valid.ravel()), N_RANDOM),
             (np.flatnonzero(band.ravel()), N_BOUNDARY),
             (np.flatnonzero(near.ravel()), N_NEAR)]
    idx = np.concatenate([rng.choice(p, min(n, len(p)), replace=False) for p, n in pools if len(p)])
    return pixel_features(img)[idx], road.ravel()[idx]


def list_images():
    return sorted(glob.glob(str(ROOT / "training" / "image_2" / "*.png")))


def bev_data(image_path):
    """Immagine BEV + GT strada + area valida, warpate con la stessa catena della pipeline."""
    import lane_pipeline as lp   # import locale: evita il giro circolare con lane_pipeline -> road_classifier

    img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    road, valid = road_gt(image_path)
    calib = lp.read_calib(ROOT / "training" / "calib" / f"{Path(image_path).stem}.txt")
    trapezoid = lp.compute_trapezoid(calib)
    img_bev, M, _ = lp.brid_eye_view(lp.apply_roi(img, trapezoid), trapezoid)
    size = (img_bev.shape[1], img_bev.shape[0])

    def warp(m):
        return cv2.warpPerspective(m.astype(np.uint8) * 255, M, size, flags=cv2.INTER_NEAREST) > 0

    return img_bev, warp(road), warp(valid)


# ---------------- addestramento ----------------
def train(val_frac=0.2):
    from sklearn.ensemble import RandomForestClassifier

    files = list_images()
    rng = np.random.default_rng(SEED)
    rng.shuffle(files)
    n_val = int(len(files) * val_frac)
    val_files, train_files = files[:n_val], files[n_val:]   # split per IMMAGINE, non per pixel

    X, y = [], []
    for f in train_files:
        img, road, valid = bev_data(f)
        for im in (img, add_shadows(img, rng)):               # originale + stessa scena con ombre
            Xi, yi = sample_pixels(im, road, valid, rng)
            X.append(Xi)
            y.append(yi)
    X, y = np.vstack(X), np.concatenate(y)
    print(f"train: {len(train_files)} immagini, {len(y)} pixel ({y.mean():.0%} strada)")

    clf = RandomForestClassifier(n_estimators=80, max_depth=16, min_samples_leaf=20,
                                 n_jobs=-1, random_state=SEED)
    clf.fit(X, y)

    val_rng = np.random.default_rng(123)
    ious = {}
    for f in val_files:
        img, road, valid = bev_data(f)
        key = Path(f).stem.split("_")[0]
        for tag, im in (("orig", img), ("ombre", add_shadows(img, val_rng))):
            pred = clf.predict(pixel_features(im)).reshape(road.shape).astype(bool)
            inter = (pred & road & valid).sum()
            union = ((pred | road) & valid).sum()
            ious.setdefault((key, tag), []).append(inter / max(union, 1))
    for (k, tag), v in sorted(ious.items()):
        print(f"val IoU strada in BEV [{k}, {tag}]: {np.mean(v):.3f}  ({len(v)} img)")

    joblib.dump(clf, MODEL_PATH, compress=3)
    print("salvato:", MODEL_PATH)
    return clf


# ---------------- inferenza nella bird-eye view ----------------
_clf = None


def load_clf():
    global _clf
    if _clf is None:
        _clf = joblib.load(MODEL_PATH)
    return _clf


def road_prob_bev(img_bird_eye, valid_bev, clf=None):
    """Probabilita' di strada per ogni pixel della BEV (utile per la diagnosi)."""
    clf = clf or load_clf()
    h, w = valid_bev.shape
    col = list(clf.classes_).index(True)
    p = clf.predict_proba(pixel_features(img_bird_eye))[:, col].reshape(h, w)
    p[~valid_bev] = 0
    return p


def road_mask_bev(img_bird_eye, valid_bev, clf=None):
    """Maschera strada (uint8 0/255) nella BEV: soglia, pulizia, componente dell'ego, buchi riempiti, bordo liscio."""
    h, w = valid_bev.shape
    mask = (road_prob_bev(img_bird_eye, valid_bev, clf) > ROAD_THR).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)     # toglie macchie isolate (erba chiara, ecc.)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)    # chiude buchi piccoli

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        seed = lab[h - 10, w // 2]                       # pixel davanti all'auto
        keep = seed if seed > 0 else 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = ((lab == keep) * 255).astype(np.uint8)

    # la strada non ha buchi: ombre, crepe e macchie interne non devono generare contorni
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, cnts, -1, 255, cv2.FILLED)
    return cv2.medianBlur(filled, 15)                    # bordo meno frastagliato


def road_edge(mask, valid_bev, margin=8):
    """Contorno della maschera strada, senza il falso bordo dovuto alle zone nere fuori immagine."""
    edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    safe = cv2.erode(valid_bev.astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    edge[safe == 0] = 0
    return edge


def chroma_edges(img_bev, valid_bev, min_grad=CHROMA_MIN, percentile=97, block=60, margin=8):
    """
    Bordi di COLORE (non di luminosita'), quasi verticali: bordo asfalto/erba anche quando la luminanza
    e' simile; i bordi d'ombra (che cambiano soprattutto la luminanza) restano sotto soglia.
    """
    ab = cv2.cvtColor(cv2.GaussianBlur(img_bev, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32)[:, :, 1:]
    gx = np.sqrt(sum(cv2.Sobel(ab[:, :, c], cv2.CV_32F, 1, 0, ksize=3) ** 2 for c in range(2)))
    gy = np.sqrt(sum(cv2.Sobel(ab[:, :, c], cv2.CV_32F, 0, 1, ksize=3) ** 2 for c in range(2)))
    g = np.hypot(gx, gy)

    safe = cv2.erode(valid_bev.astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)) > 0
    out = np.zeros(g.shape, np.uint8)
    for c in range(0, g.shape[1], block):
        ok = safe[:, c:c + block]
        if ok.any():
            thr = max(np.percentile(g[:, c:c + block][ok], percentile), min_grad)
            out[:, c:c + block] = (g[:, c:c + block] > thr) * 255
    out[~safe | (gx < 1.5 * gy)] = 0          # solo bordi quasi verticali (gradiente orizzontale)
    return out


if __name__ == "__main__":
    train()