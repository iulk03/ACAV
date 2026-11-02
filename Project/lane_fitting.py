import matplotlib.pyplot as plt
import numpy as np
import cv2
from lane_pipeline import X_RANGE, Z_RANGE, PX_PER_M

# ---------------- parametri RANSAC (Ottimizzati per Canny) ----------------
MARGIN_PX = 6               # esclude i pixel troppo vicini ai bordi
INLINER_TRESHOLD_M = 0.12   # soglia in metri per essere inlier
MIN_SEPARATION_M = 4.0      # distanza minima in z tra i due punti
SUCCESS_PROBABILITY = 0.99  #
MAX_ITER = 500              #
MIN_INLIERS = 25            # ridotto per adattarsi al bordo Canny a 1 pixel
MAX_SLOPE = 0.5             # pendenza massima (x = a*z + b)

# ---------------- parametri di scelta della coppia ----------------
N_CANDIDATES = 5            # quante rette candidate cercare
Z_REF = 0.5 * (Z_RANGE[0] + Z_RANGE[1])   # z di riferimento per la posizione laterale
LANE_W_MIN = 2.5            # larghezza minima plausibile della corsia (m)
LANE_W_MAX = 4.5            # larghezza massima plausibile (m)
SLOPE_DELTA_MAX = 0.1       # differenza massima di pendenza tra le due rette
W_CENTER = 100.0            # peso della penalità centro auto
LANE_WIDTH_M = 3.2          # larghezza di default per il ripiego

# Memoria per lo smoothing temporale
prev_left_line = None
prev_right_line = None


def smooth_line(current_line, prev_line, alpha=0.6):
    """Filtro esponenziale per evitare scatti visivi tra frame successivi."""
    if prev_line is None or current_line is None:
        return current_line
    smoothed = dict(current_line)
    smoothed["a"] = alpha * current_line["a"] + (1 - alpha) * prev_line["a"]
    smoothed["b"] = alpha * current_line["b"] + (1 - alpha) * prev_line["b"]
    return smoothed

def reset_tracking():
    """Clear temporal smoothing state (call between unrelated images)."""
    global prev_left_line, prev_right_line
    prev_left_line = prev_right_line = None

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
    return line["a"] * z + line["b"]


def find_candidates(points, n=N_CANDIDATES, threshold=INLINER_TRESHOLD_M):
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
    larg = x_at(r) - x_at(l)
    if not (LANE_W_MIN < larg < LANE_W_MAX):
        return -np.inf
    if abs(l["a"] - r["a"]) > SLOPE_DELTA_MAX:
        return -np.inf
    centro = 0.5 * (x_at(l) + x_at(r))
    penalita = W_CENTER * abs(centro)
    return int(l["inliers"].sum()) + int(r["inliers"].sum()) - penalita


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
    est = dict(a=line["a"], b=None, inliers=None, n_iter=0, stimata=True)
    if x_at(line) > 0:
        est["b"] = line["b"] - width
        return est, line
    est["b"] = line["b"] + width
    return line, est


# ---------------- disegno e prospettiva ----------------
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


# ---------------- funzione principale di fitting ----------------
def lane_fitting(img_tresh, img, M_inv):
    global prev_left_line, prev_right_line

    pts_dict = points_in_meters(img_tresh)
    pts = np.column_stack((pts_dict["x"], pts_dict["z"]))

    cands = find_candidates(pts)
    pair = best_pair(cands)

    ransac_sx = ransac_dx = None
    stimata = False
    if pair is not None:
        ransac_sx, ransac_dx = pair
    elif cands:
        best = max(cands, key=lambda c: int(c["inliers"].sum()))
        ransac_sx, ransac_dx = missing_side(best)
        stimata = True

    # Smoothing temporale
    ransac_sx = smooth_line(ransac_sx, prev_left_line)
    ransac_dx = smooth_line(ransac_dx, prev_right_line)

    prev_left_line = ransac_sx
    prev_right_line = ransac_dx

    out = dict(points=pts_dict, pts=pts, candidates=cands,
               ransac_sx=ransac_sx, ransac_dx=ransac_dx, stimata=stimata)
    if ransac_sx and ransac_dx:
        out["overlay"] = draw_line(img, M_inv, ransac_sx, ransac_dx)
    return out


# ---------------- funzione di debug visivo BEV ----------------
def plot_bev_debug(img_tresh, fitting_out, out_path=None):
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.imshow(
        img_tresh,
        cmap="gray",
        origin="upper",
        extent=(X_RANGE[0], X_RANGE[1], Z_RANGE[0], Z_RANGE[1]),
        aspect="auto",
    )
    ax.axvline(0, color="yellow", linestyle="--", linewidth=1, label="Centro veicolo")

    pts = fitting_out["pts"]
    for label, line_key, color in (("Sinistra", "ransac_sx", "red"), ("Destra", "ransac_dx", "cyan")):
        line = fitting_out[line_key]
        if line is None:
            ax.plot([], [], color=color, label=f"{label}: retta non trovata")
            continue

        if line.get("inliers") is not None:
            inl = pts[line["inliers"]]
            ax.scatter(inl[:, 0], inl[:, 1], s=5, color=color,
                       label=f"Inlier {label.lower()} ({len(inl)})")
            z_values = np.linspace(inl[:, 1].min(), inl[:, 1].max(), 100) if len(inl) > 0 else np.linspace(Z_RANGE[0], Z_RANGE[1], 100)
            style = "-"
            name = f"Retta {label.lower()}"
        else:
            z_values = np.linspace(Z_RANGE[0], Z_RANGE[1], 100)
            style = "--"
            name = f"Retta {label.lower()} (STIMATA)"
        ax.plot(line["a"] * z_values + line["b"], z_values,
                color=color, linewidth=2, linestyle=style, label=name)

    ax.set(
        xlabel="x laterale (m)",
        ylabel="z in avanti (m)",
        title="Bird-Eye View: pixel candidati e inlier RANSAC (Canny)",
        xlim=X_RANGE,
        ylim=Z_RANGE,
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()



def print_candidate_stats(out):
    """Print the numbers that decide snapping and lane selection."""
    for k, c in enumerate(out["candidates"]):
        print(f"cand {k}: x@mid={x_at(c):+.2f} a={c['a']:+.2f} cov={c['cov_m']:.1f}m "
                f"paint={c['paint']:.2f} ridge={c['ridge_m']:.1f}m inl={int(c['inliers'].sum())} ego={c['ego_m']:.1f}m sim={c.get('sim_m', 0.0):.1f}m score: {line_score(c, True):.2f} ")
    for l, r, _ in out["lanes"]:
        for name, ln in (("left", l), ("right", r)):
            print(f"ego {name}: x@mid={x_at(ln):+.2f} edge={ln.get('edge', False)} snapped={ln.get('snapped', False)}")