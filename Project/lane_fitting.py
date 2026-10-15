import matplotlib.pyplot as plt
import numpy as np
import cv2
from lane_pipeline import X_RANGE, Z_RANGE, PX_PER_M

MARGIN_PX = 6 #per escludere i pixel troppo vicini ai bordi 
INLINER_TRESHOLD_M = 0.20 #punto considerato vicino alla retta
MIN_SEPARATION_M = 4.0
SUCCESS_PROBABILITY = 0.99 #
MAX_ITER = 500 #limite massimo di iterazioni
MIN_INLIERS = 50 #una retta deve avere almeno 50 punti vicini per essere considerata tale
MAX_SLOPE = 0.3
#LANE_WIDTH_M = (2.5, 4.5) NON LO STIAMO USANDO
SLOPE_DELTA_MAX = 0.1 # max difference in slope between 2 lines 
    #DA CAPIRE LA DOCCIA COSA DELL'INCLINAZIONE

#da immagine bianca e nera a coordinate della strada 
def points_in_meters(img_tresh, px_per_m=PX_PER_M, x_range=X_RANGE, z_range=Z_RANGE, margin_px=MARGIN_PX):
    height, width = img_tresh.shape
    row,column = np.nonzero(img_tresh)
    ok = (row >= margin_px) & (row < height - margin_px) & (column >= margin_px) & (column < width - margin_px)
    row, column = row[ok], column[ok]
    x = x_range[0] + column / px_per_m #posizione laterale in metri
    z = z_range[1] - row / px_per_m #distanza avanti in metri
    return dict(x=x, z=z) #dizionario con tutti i pixel bianchi rimasti dopo il filtraggo (linee + rumore)

def div_lft_rgt(x, z):
    """Divide i punti in due gruppi: a sinistra e a destra dell'auto."""
    lft = x < 0
    rgt = x > 0
    return np.column_stack((x[lft], z[lft])), np.column_stack((x[rgt], z[rgt])) #ogni riga è un singolo punto

def n_iteration(p,w,n_points=2):
    w_n = w ** n_points
    if(w_n >= 1.0):
        return 1
    if(w_n <= 0.0):
        return MAX_ITER
    return int(np.ceil(np.log(1.0 - p) / np.log(1.0 - w_n)))

def line_distance(points, a, b):
    return np.abs(points[:, 0] - (a * points[:, 1] + b)) / np.sqrt(a**2 + 1)

def line_fit(p1, p2, separation=MIN_SEPARATION_M, slope_delta=SLOPE_DELTA_MAX):
    (x1,z1), (x2,z2) = p1, p2
    dz = z2 - z1
    if abs(dz) < separation:
        return None
    a = (x2 - x1) / dz
    if abs(a) > slope_delta:
        return None
    return a, x1 - a * z1

#eseguita una volta per lato, riceve i punti x e z e prova a descriverli con una retta x=az+b
def ransac(points, threshold=INLINER_TRESHOLD_M, min_inliers=MIN_INLIERS, max_iter=MAX_ITER, p=SUCCESS_PROBABILITY, max_slope=MAX_SLOPE, sep_min=MIN_SEPARATION_M, seed=0):
    N = len(points)
    if N < max(2, min_inliers):
        return None #se ho meno di due punti o cmq meno di quelli che abbiamo deciso per formare una retta restituisce None
    range = np.random.default_rng(seed)
    best, n_best, best_mask = None, 0, None
    k, n_iter, attempts = max_iter, 0, 0
    while n_iter < min(k, max_iter) and attempts < 50*max_iter:
        attempts += 1
        i,j = range.integers(0, N, size=2)
        line = line_fit(points[i], points[j], separation=sep_min, slope_delta=max_slope) #costuisce una retta
        if line is not None:
            a,b = line
            n_iter += 1
            inliers = line_distance(points, a, b) <= threshold #distanza di tutti i punti dalla retta 
            n_inliers = int(inliers.sum())
            if n_inliers > n_best: #se la retta ha più inliner di quella trovata viene conservata 
                best, n_best, best_mask = (a,b), n_inliers, inliers
                k = n_iteration(p, n_best/N)
    if best is None or n_best < min_inliers:
        return None
    a,b = np.polyfit(points[best_mask, 1], points[best_mask, 0], 1)
    #rifa i controlli dopo aver calcolato la retta 
    if abs(a) > max_slope:
        return None
    best_mask = line_distance(points, a, b) <= threshold
    if int(best_mask.sum()) < min_inliers:
        return None
    return dict(a=float(a),b=float(b), inliers=best_mask, n_iter=n_iter)

def line_on_image(line, M_inv, x_range=X_RANGE, z_range=Z_RANGE, px_per_m=PX_PER_M):
    a,b = line["a"], line["b"]
    z = np.linspace(z_range[0], z_range[1], 100)
    #100 posizioni lungo la retta considerazno z range 
    #POSSIBILE PROBLEMA: IO DISEGNO SU TUTTO ZRANGE.. NON SOLO SUI PIXEL BIANCHI 
    x = a * z + b
    col = (x - x_range[0]) * px_per_m
    row = (z_range[1] - z) * px_per_m
    points_px = np.column_stack((col, row)).astype(np.float32).reshape(-1, 1, 2)
    #trasforma in coordinate viste dall'alto e applica M_inv per riportarle sull'immagine
    return cv2.perspectiveTransform(points_px, M_inv).reshape(-1, 2)

def draw_line(img, M_inv, line_lft, line_rgt, alpha=0.35):
    lf = line_on_image(line_lft, M_inv)
    rg = line_on_image(line_rgt, M_inv)
    area = np.vstack((lf, rg[::-1])).astype(np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [area], (0, 255, 0))
    out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.polylines(out, [lf.astype(np.int32)], False, (255, 0, 0), 3)
    cv2.polylines(out, [rg.astype(np.int32)], False, (0, 0, 255), 3)
    return out

def lane_fitting(img_tresh, img, M_inv):
    points = points_in_meters(img_tresh)
    pts_sx, pts_dx = div_lft_rgt(points["x"], points["z"])
    out = dict(points=points, pts_sx=pts_sx, pts_dx=pts_dx,
               ransac_sx=ransac(pts_sx), ransac_dx=ransac(pts_dx))
    if out["ransac_sx"] and out["ransac_dx"]:
        out["overlay"] = draw_line(img, M_inv, out["ransac_sx"], out["ransac_dx"])
    return out


def plot_bev_debug(img_tresh, fitting_out):
    """Mostra i pixel candidati e gli inlier RANSAC nelle coordinate BEV."""
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.imshow(
        img_tresh,
        cmap="gray",
        origin="upper",
        extent=(X_RANGE[0], X_RANGE[1], Z_RANGE[0], Z_RANGE[1]),
        aspect="auto",
    )
    ax.axvline(0, color="yellow", linestyle="--", linewidth=1, label="Centro veicolo")

    for label, points_key, line_key, color in (
        ("Sinistra", "pts_sx", "ransac_sx", "red"),
        ("Destra", "pts_dx", "ransac_dx", "cyan"),
    ):
        points = fitting_out[points_key]
        line = fitting_out[line_key]
        if line is None:
            ax.plot([], [], color=color, label=f"{label}: retta non trovata")
            continue

        inliers = points[line["inliers"]]
        ax.scatter(
            inliers[:, 0],
            inliers[:, 1],
            s=5,
            color=color,
            label=f"Inlier {label.lower()} ({len(inliers)})",
        )
        z_values = np.linspace(inliers[:, 1].min(), inliers[:, 1].max(), 100)
        ax.plot(
            line["a"] * z_values + line["b"],
            z_values,
            color=color,
            linewidth=2,
            label=f"Retta {label.lower()}",
        )

    ax.set(
        xlabel="x laterale (m)",
        ylabel="z in avanti (m)",
        title="Bird-Eye View: pixel candidati e inlier RANSAC",
        xlim=X_RANGE,
        ylim=Z_RANGE,
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()
