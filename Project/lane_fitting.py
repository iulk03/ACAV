import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cv2
from lane_pipeline import X_RANGE, Z_RANGE, PX_PER_M

# ---------------- parametri RANSAC ----------------
MARGIN_PX = 6               # esclude i pixel troppo vicini ai bordi
INLINER_TRESHOLD_M = 0.20   # distanza massima per essere inlier
MIN_SEPARATION_M = 3.0      # distanza minima in z tra i due punti che definiscono una retta
SUCCESS_PROBABILITY = 0.99
MAX_ITER = 500
MIN_INLIERS = 100
MAX_SLOPE = 0.5             # pendenza massima di una singola retta (x = a*z + b)
MERGE_DIST_M = 1.0

# ---------------- parametri di scelta della coppia ----------------
N_CANDIDATES = 10            # quante rette candidate cercare (ogni linea bianca ha 2 bordi)
Z_REF = 0.5 * (Z_RANGE[0] + Z_RANGE[1])   # z a cui si misura la posizione laterale (dentro il range)
LANE_W_MIN = 2.5            # larghezza minima plausibile della corsia (m)
LANE_W_MAX = 9            # larghezza massima plausibile (m)
SLOPE_DELTA_MAX = 0.5       # differenza massima di pendenza tra le due rette
W_CENTER = 100.0            # peso della penality' "auto lontana dal centre della coppia" (in pixel/metro)
W_CONTRAST = 0.0            # TO BE TUNED (0-255)
LANE_WIDTH_M = 3.2          # larghezza di default per il ripiego (un solo lato trovato)

CAND_COLORS = {
    "red":          (255, 0, 0),
    "cyan":         (0, 255, 255),
    "lime":         (0, 255, 0),
    "magenta":      (255, 0, 255),
    "orange":       (255, 165, 0),
    "dodgerblue":   (30, 144, 255),
    "gold":         (255, 215, 0),
    "deeppink":     (255, 20, 147),
    "white":        (255, 255, 255),
    "mediumpurple": (147, 112, 219),
    }

# ---------------- da immagine binaria a coordinate strada ----------------
def points_in_meters(img_tresh, px_per_m=PX_PER_M, x_range=X_RANGE, z_range=Z_RANGE, margin_px=MARGIN_PX):
    height, width = img_tresh.shape
    row, column = np.nonzero(img_tresh)
    ok = (row >= margin_px) & (row < height - margin_px) & (column >= margin_px) & (column < width - margin_px)
    row, column = row[ok], column[ok]
    x = x_range[0] + column / px_per_m
    z = z_range[1] - row / px_per_m
    return dict(x=x, z=z)


# ---------------- RANSAC (una retta) ----------------
def n_iteration(p, w, n_points=2):
    w_n = w ** n_points
    if w_n >= 1.0:
        return 1
    if w_n <= 0.0:
        return MAX_ITER
    return int(np.ceil(np.log(1.0 - p) / np.log(1.0 - w_n)))


def line_distance(points, a, b):
    return np.abs(points[:, 0] - (a * points[:, 1] + b)) / np.sqrt(a ** 2 + 1)


def line_fit(p1, p2, separation=MIN_SEPARATION_M, max_slope=MAX_SLOPE):
    (x1, z1), (x2, z2) = p1, p2
    dz = z2 - z1
    if abs(dz) < separation:
        return None
    a = (x2 - x1) / dz
    if abs(a) > max_slope:
        return None
    return a, x1 - a * z1


def ransac(points, threshold=INLINER_TRESHOLD_M, min_inliers=MIN_INLIERS, max_iter=MAX_ITER,
            p=SUCCESS_PROBABILITY, max_slope=MAX_SLOPE, sep_min=MIN_SEPARATION_M, seed=0):
    N = len(points)
    if N < max(2, min_inliers):
        return None
    rng = np.random.default_rng(seed)
    best, n_best, best_mask = None, 0, None
    k, n_iter, attempts = max_iter, 0, 0
    while n_iter < min(k, max_iter) and attempts < 50 * max_iter:
        attempts += 1
        i, j = rng.integers(0, N, size=2)
        line = line_fit(points[i], points[j], separation=sep_min, max_slope=max_slope)
        if line is not None:
            a, b = line
            n_iter += 1
            inliers = line_distance(points, a, b) <= threshold
            n_inliers = int(inliers.sum())
            if n_inliers > n_best:
                best, n_best, best_mask = (a, b), n_inliers, inliers
                k = n_iteration(p, n_best / N)
    if best is None or n_best < min_inliers:
        return None
    a, b = np.polyfit(points[best_mask, 1], points[best_mask, 0], 1)
    if abs(a) > max_slope:
        return None
    best_mask = line_distance(points, a, b) <= threshold
    if int(best_mask.sum()) < min_inliers:
        return None
    return dict(a=float(a), b=float(b), inliers=best_mask, n_iter=n_iter)


# ---------------- candidati multipli e scelta della coppia ----------------
def x_at(line, z=Z_REF):
    """Posizione laterale della line a una z dentro il range (non a z = 0, che e' estrapolata)."""
    return line["a"] * z + line["b"]


def find_candidates(points, n=N_CANDIDATES, threshold=INLINER_TRESHOLD_M):
    """
    RANSAC sequenziale su TUTTI i punti: trova la line migliore, rimuove i punti vicini
    (banda 2*threshold, cosi' non ripesca l'altro bordo della stessa linea), e ripete.
    La maschera 'inliers' di ogni candidato e' riferita all'array originale.
    """
    cands = []
    remaining = np.arange(len(points))
    for _ in range(n):
        if len(remaining) < MIN_INLIERS:
            break
        sub = points[remaining]
        r = ransac(sub)
        if r is None:
            break
        full_mask = np.zeros(len(points), dtype=bool)
        full_mask[remaining[r["inliers"]]] = True
        r["inliers"] = full_mask
        cands.append(r)
        near = line_distance(sub, r["a"], r["b"]) <= 2 * threshold
        remaining = remaining[~near]
    return cands


def pair_score(l, r):
    """Punteggio di una coppia (sinistra, destra). -inf se non e' una corsia plausibile."""
    larg = x_at(r) - x_at(l)
    if not (LANE_W_MIN < larg < LANE_W_MAX):
        return -np.inf
    if abs(l["a"] - r["a"]) > SLOPE_DELTA_MAX:
        return -np.inf
    centre = 0.5 * (x_at(l) + x_at(r))
    penality = W_CENTER * abs(centre)   # morbida: un cambio corsia puo' ancora vincere
    contrast = W_CONTRAST * (l["contrast"] + r["contrast"])
    return int(l["inliers"].sum()) + int(r["inliers"].sum()) - penality + contrast


def best_pair(cands):
    best, best_s = None, -np.inf
    for l in cands:
        for r in cands:
            if l is r or x_at(l) >= x_at(r):
                continue
            s = pair_score(l, r)
            if s > best_s:
                best, best_s = (l, r), s
    return best


def missing_side(line, width=LANE_WIDTH_M):
    """
    Ripiego: una sola line affidabile. Se sta a destra del centre e' il bordo destro,
    altrimenti il sinistro. L'altro lato e' STIMATO (stessa pendenza, spostato di 'width').
    """
    est = dict(a=line["a"], b=None, inliers=None, n_iter=0, estimate=True)
    if x_at(line) > 0:
        est["b"] = line["b"] - width
        return est, line          # (sinistra estimate, destra misurata)
    est["b"] = line["b"] + width
    return line, est              # (sinistra misurata, destra estimate)


# ---------------- disegno ----------------
def line_on_image(line, M_inv, x_range=X_RANGE, z_range=Z_RANGE, px_per_m=PX_PER_M):
    a, b = line["a"], line["b"]
    z = np.linspace(z_range[0], z_range[1], 100)
    x = a * z + b
    col = (x - x_range[0]) * px_per_m
    row = (z_range[1] - z) * px_per_m
    points_px = np.column_stack((col, row)).astype(np.float32).reshape(-1, 1, 2)
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

def ridge_contrast(gray, line, px_per_m=PX_PER_M, x_range=X_RANGE, z_range=Z_RANGE):
    h, w = gray.shape
    rows, cols = np.mgrid[0:h, 0:w]
    x = x_range[0] + cols / px_per_m
    z = z_range[1] - rows / px_per_m
    d = np.abs(x - (line["a"] * z + line["b"])) / np.sqrt(line["a"] ** 2 + 1)
    centre = gray[d < 0.10]
    sides = gray[(d > 0.20) & (d < 0.40)]
    if centre.size == 0 or sides.size == 0:
        return 0.0
    return float(centre.mean() - sides.mean())

# ---------------- funzione principale ----------------
def lane_fitting(pipe, img):
    pts_dict = points_in_meters(pipe["thresh"])
    pts = np.column_stack((pts_dict["x"], pts_dict["z"]))

    cands = find_candidates(pts)
    edge_l = road_edge_candidate(pipe["road"], pipe["valid"], -1)
    edge_r = road_edge_candidate(pipe["road"], pipe["valid"], +1)
    lanes = build_lanes(cands, edge_l, edge_r)
    ego = next((ln for ln in lanes if ln[2]), None)

    out = dict(points=pts_dict, pts=pts, candidates=cands, edges=[e for e in (edge_l, edge_r) if e], lanes=lanes,
                ransac_sx=ego[0] if ego else None, ransac_dx=ego[1] if ego else None, estimate=False)
    if lanes:
        out["overlay"] = draw_lanes(img, pipe["M_inv"], lanes)
    return out


def plot_bev_debug(img_tresh, fitting_out, out_path):
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.imshow(
        img_tresh,
        cmap="gray",
        origin="upper",
        extent=(X_RANGE[0], X_RANGE[1], Z_RANGE[0], Z_RANGE[1]),
        aspect="auto",
    )
    ax.axvline(0, color="yellow", linestyle="--", linewidth=1, label="vehicle centre")

    pts = fitting_out["pts"]
    for label, line_key, color in (("Left", "ransac_sx", "red"), ("Right", "ransac_dx", "cyan")):
        line = fitting_out[line_key]
        if line is None:
            ax.plot([], [], color=color, label=f"{label}: line not found")
            continue

        if line.get("inliers") is not None:
            inl = pts[line["inliers"]]
            ax.scatter(inl[:, 0], inl[:, 1], s=5, color=color, label=f"Inlier {label.lower()} ({len(inl)})")
            z_values = np.linspace(inl[:, 1].min(), inl[:, 1].max(), 100)
            style = "-"
            name = f"Line {label.lower()}"
        else:
            name = f"Line {label.lower()} (road edge)" if line.get("edge") else f"Line {label.lower()} (estimate)"
            # lato stimato dalla larghezza di default: tratteggiato su tutto il range
            z_values = np.linspace(Z_RANGE[0], Z_RANGE[1], 100)
            style = "--"
            name = f"Line {label.lower()} (estimate)"
        ax.plot(line["a"] * z_values + line["b"], z_values,
                color=color, linewidth=2, linestyle=style, label=name)

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
    #plt.show()
    # save the image in the folder debug_images
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def lane_mask(shape, line_lft, line_rgt, M_inv):
    lf_px = line_on_image(line_lft, M_inv)
    rg_px = line_on_image(line_rgt, M_inv)
    area = np.vstack((lf_px, rg_px[::-1])).astype(np.int32)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [area], 1)
    return mask.astype(bool)

def cand_rgb(k):   # tuple RGB 0-255 per cv2 (l'immagine e' in RGB)
    return tuple(int(255 * v) for v in mcolors.to_rgb(list(CAND_COLORS.keys())[k % len(CAND_COLORS)]))

def save_candidates(img, img_tresh, pts, cands, M_inv, out_bev, out_img):
    # 1) bird-eye view con tutti i candidati
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.imshow(img_tresh, cmap="gray", origin="upper", extent=(X_RANGE[0], X_RANGE[1], Z_RANGE[0], Z_RANGE[1]), aspect="auto")
    ax.axvline(0, color="yellow", linestyle="--", linewidth=1)
    z = np.linspace(Z_RANGE[0], Z_RANGE[1], 100)
    for k, c in enumerate(cands):
        inl = pts[c["inliers"]] if c.get("inliers") is not None else pts[:0]
        col = list(CAND_COLORS.keys())[k % len(CAND_COLORS)]
        inl = pts[c["inliers"]]
        ax.scatter(inl[:, 0], inl[:, 1], s=4, color=col)
        ax.plot(c["a"] * z + c["b"], z, color=col, linewidth=2,
                label=f"cand {k}: a={c['a']:+.2f} b={c['b']:+.2f} inl={len(inl)}")
    ax.set(xlabel="x laterale (m)", ylabel="z in avanti (m)", title="Candidati RANSAC", xlim=X_RANGE, ylim=Z_RANGE)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_bev, dpi=150)
    plt.close(fig)

    # 2) immagine originale con tutti i candidati riproiettati
    vis = img.copy()
    for k, c in enumerate(cands):
        pts_px = line_on_image(c, M_inv).astype(np.int32)
        cv2.polylines(vis, [pts_px], False, list(CAND_COLORS.values())[k % len(CAND_COLORS)], 3)
    cv2.imwrite(out_img, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def road_edge_candidate(road, valid, direction):
    p = road_edge_points(road, valid, direction)
    if len(p) < 40:
        return None
    r = ransac(p, threshold=0.15, min_inliers=40)
    if r is None:
        return None
    r["inliers"] = None        # i punti del bordo non stanno nell'array 'pts' del gradiente
    r["edge"] = True
    return r


def _rank(c):
    return (1, 0) if c.get("edge") else (0, int(c["inliers"].sum()))


def filter_parallel(cands, edges):
    if not cands:
        return []
    if edges:
        a_ref = float(np.mean([e["a"] for e in edges]))     # la direzione la danno i bordi
    else:
        a = np.array([c["a"] for c in cands])
        w = np.array([c["inliers"].sum() for c in cands], float)
        o = np.argsort(a); cw = np.cumsum(w[o])
        a_ref = a[o][np.searchsorted(cw, cw[-1] / 2)]       # mediana pesata
    return [c for c in cands if abs(c["a"] - a_ref) < SLOPE_DELTA_MAX]


def merge_close(lines):
    out = []
    for c in sorted(lines, key=x_at):
        if out and x_at(c) - x_at(out[-1]) < MERGE_DIST_M:
            if _rank(c) > _rank(out[-1]):                   # il bordo strada vince sempre
                out[-1] = c
        else:
            out.append(c)
    return out


def build_lanes(cands, left_edge, right_edge):
    edges = [e for e in (left_edge, right_edge) if e]
    painted = filter_parallel(cands, edges)
    if left_edge:
        painted = [c for c in painted if x_at(c) > x_at(left_edge) + MERGE_DIST_M]
    if right_edge:
        painted = [c for c in painted if x_at(c) < x_at(right_edge) - MERGE_DIST_M]
    lines = merge_close(painted + edges)
    return [[l, r, x_at(l) < 0 < x_at(r)] for l, r in zip(lines[:-1], lines[1:])]


def draw_lanes(img, M_inv, lanes, alpha=0.35):
    overlay = img.copy()
    for l, r, ego in lanes:
        area = np.vstack((line_on_image(l, M_inv), line_on_image(r, M_inv)[::-1])).astype(np.int32)
        cv2.fillPoly(overlay, [area], (0, 255, 0) if ego else (255, 0, 0))
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)