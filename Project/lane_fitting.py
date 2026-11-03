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
LANE_W_MIN = 2.0            # larghezza minima plausibile della corsia (m)
LANE_W_MAX = 5.5            # max plausible lane width (m): wide intersections
SLOPE_DELTA_MAX = 0.1       # differenza massima di pendenza tra le due rette
W_CENTER = 20.0             # weak penalty for a lane not centred on the car
W_WIDTH = 40.0              # penalty per metre away from LANE_WIDTH_M
LANE_WIDTH_M = 3.2          # larghezza di default per il ripiego
MASK_EDGE_MARGIN_PX = 4     # border columns closer than this to the BEV side are fake (clipped by the ROI)
MASK_MIN_ROWS = 40          # min number of rows with a valid border point


CAR_TOL_M = 0.3             # car (x=0) must lie between the two lines, within this tolerance
W_SUPPORT_CAP = 400.0       # cap on the weighted inlier support of a single line
W_GUIDE = 80.0              # penalty per metre of disagreement with the guide border (lane guide only)
GUIDE_TOL_M = 0.6           # max distance for a candidate to be considered "matching" a guide border


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


def ransac(points, threshold=INLINER_TRESHOLD_M, min_inliers=MIN_INLIERS, max_iter=MAX_ITER, p=SUCCESS_PROBABILITY, max_slope=MAX_SLOPE, sep_min=MIN_SEPARATION_M, seed=0):
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


def x_car(line):
    """Lateral position of the line at z=0 (where the car is), not at Z_REF: matters when the camera is yawed."""
    return line["b"]


def near_weights(pts):
    """Per-point weight: `gain` at the closest z, linearly down to 1.0 at the farthest z."""
    t = (Z_RANGE[1] - pts[:, 1]) / (Z_RANGE[1] - Z_RANGE[0])      # 1 near the camera, 0 far
    return 1.0 + 2.0 * np.clip(t, 0.0, 1.0)


def find_candidates(points, n=N_CANDIDATES, threshold=INLINER_TRESHOLD_M):
    cands = []
    w = near_weights(points)
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
        r["w_inl"] = float(w[full_mask].sum())
        cands.append(r)
        near = line_distance(sub, r["a"], r["b"]) <= 2 * threshold
        remaining = remaining[~near]
    return cands


def pair_score(l, r, gl=None, gr=None):
    xl, xr = x_at(l), x_at(r)
    larg = xr - xl                                     # width: same at any z for ~parallel lines
    if not (LANE_W_MIN < larg < LANE_W_MAX):
        return -np.inf
    if abs(l["a"] - r["a"]) > SLOPE_DELTA_MAX:
        return -np.inf
    cl, cr = x_car(l), x_car(r)                        
    if not (cl < CAR_TOL_M and cr > -CAR_TOL_M):
        return -np.inf
    support = min(l["w_inl"], W_SUPPORT_CAP) + min(r["w_inl"], W_SUPPORT_CAP)
    penalty = W_CENTER * abs(0.5 * (cl + cr)) + W_WIDTH * abs(larg - LANE_WIDTH_M)
    if gl is not None:
        penalty += W_GUIDE * abs(xl - x_at(gl))
    if gr is not None:
        penalty += W_GUIDE * abs(xr - x_at(gr))
    return support - penalty


def best_pair(cands, gl=None, gr=None):
    best, best_s = None, -np.inf
    for l in cands:
        for r in cands:
            if l is r or x_at(l) >= x_at(r):
                continue
            s = pair_score(l, r, gl, gr)
            if s > best_s:
                best, best_s = (l, r), s
    return best


def pick_anchor(cands, gl, gr):
    """Candidate used when no valid pair exists: the one closest to a guide border, else the strongest."""
    refs = [g for g in (gl, gr) if g is not None]
    if refs:
        scored = [(min(abs(x_at(c) - x_at(g)) for g in refs), c) for c in cands]
        scored = [t for t in scored if t[0] < GUIDE_TOL_M]
        if scored:
            return min(scored, key=lambda t: t[0])[1]
    return max(cands, key=lambda c: c["w_inl"])


def missing_side(line, width=LANE_WIDTH_M):
    est = dict(a=line["a"], b=None, inliers=None, n_iter=0, stimata=True)
    if x_at(line) > 0:
        est["b"] = line["b"] - width
        return est, line
    est["b"] = line["b"] + width
    return line, est

def mask_border_line(mask, side, px_per_m=PX_PER_M, x_range=X_RANGE, z_range=Z_RANGE):
    """Fit x = a*z + b on the left/right border of a binary BEV mask (one point per row)."""
    h, w = mask.shape
    pts = []
    for r in range(MARGIN_PX, h - MARGIN_PX):
        cols = np.flatnonzero(mask[r])
        if len(cols) == 0:
            continue
        c = cols[0] if side == "left" else cols[-1]
        if c <= MASK_EDGE_MARGIN_PX or c >= w - 1 - MASK_EDGE_MARGIN_PX:
            continue
        pts.append((x_range[0] + c / px_per_m, z_range[1] - r / px_per_m))
    if len(pts) < MASK_MIN_ROWS:
        return None
    return ransac(np.array(pts), min_inliers=MASK_MIN_ROWS)


def guide_borders(guide):
    """(left, right) border lines of the guide mask. Only if the guide is the ego-lane mask, else (None, None)."""
    if guide is None or guide.get("mask") is None or guide.get("source") != "lane":
        return None, None
    return mask_border_line(guide["mask"], "left"), mask_border_line(guide["mask"], "right")


def missing_side_from_guide(line, guide):
    """Other lane border taken from the mask edge. Returns (left, right) or None."""
    if guide is None or guide.get("mask") is None:
        return None
    known_is_right = x_at(line) > 0
    edge = mask_border_line(guide["mask"], "left" if known_is_right else "right")
    if edge is None:
        return None
    width = x_at(line) - x_at(edge) if known_is_right else x_at(edge) - x_at(line)
    if not (LANE_W_MIN < width < LANE_W_MAX) or abs(edge["a"] - line["a"]) > SLOPE_DELTA_MAX:
        return None
    est = dict(a=edge["a"], b=edge["b"], inliers=None, n_iter=0, stimata=True, source=guide["source"])
    return (est, line) if known_is_right else (line, est)


def _index(cands, c):
    return next((i for i, x in enumerate(cands) if x is c), None)   # identity: dicts hold numpy arrays


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
def lane_fitting(img_tresh, img, M_inv, guide=None):

    pts_dict = points_in_meters(img_tresh)
    pts = np.column_stack((pts_dict["x"], pts_dict["z"]))

    gl, gr = guide_borders(guide)
    cands = find_candidates(pts)
    pair = best_pair(cands, gl, gr)

    ransac_sx = ransac_dx = None
    stimata = False
    sel = dict(left=None, right=None)
    if pair is not None:
        ransac_sx, ransac_dx = pair
        sel = dict(left=_index(cands, pair[0]), right=_index(cands, pair[1]))
    elif cands:
        best = pick_anchor(cands, gl, gr)
        ransac_sx, ransac_dx = missing_side_from_guide(best, guide) or missing_side(best)
        sel["right" if x_at(best) > 0 else "left"] = _index(cands, best)
        stimata = True

    out = dict(points=pts_dict, pts=pts, candidates=cands, ransac_sx=ransac_sx, ransac_dx=ransac_dx, stimata=stimata, sel=sel, guide=guide)
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
            ax.scatter(inl[:, 0], inl[:, 1], s=5, color=color, label=f"Inlier {label.lower()} ({len(inl)})")
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


def plot_candidates_thresh(img_tresh, out, out_path=None):
    """Same look as plot_bev_debug (black thresh image, inlier scatter, fitted line), but for EVERY RANSAC candidate.
    Selected candidates: solid line + role in the legend. Unselected: dotted line."""
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.imshow(img_tresh, cmap="gray", origin="upper", extent=(X_RANGE[0], X_RANGE[1], Z_RANGE[0], Z_RANGE[1]), aspect="auto")
    ax.axvline(0, color="yellow", linestyle="--", linewidth=1, label="Vehicle center")

    guide = out.get("guide")
    if guide is not None and guide.get("mask") is not None:
        ax.contour((guide["mask"] > 0).astype(float), levels=[0.5], colors="white", linewidths=1.5, extent=(X_RANGE[0], X_RANGE[1], Z_RANGE[0], Z_RANGE[1]), origin="upper")
        ax.plot([], [], color="white", label=f"guide mask ({guide['source']})")

    pts = out["pts"]
    role = {v: k for k, v in out["sel"].items() if v is not None}
    for k, c in enumerate(out["candidates"]):
        col = cmap(k % 10)
        inl = pts[c["inliers"]]
        ax.scatter(inl[:, 0], inl[:, 1], s=5, color=col)
        z = np.linspace(inl[:, 1].min(), inl[:, 1].max(), 100)
        tag = f" -> {role[k].upper()}" if k in role else ""
        ax.plot(c["a"] * z + c["b"], z, color=col, linewidth=2,
                linestyle="-" if k in role else ":",
                label=f"#{k} x@mid={x_at(c):+.2f} a={c['a']:+.2f} n={len(inl)}{tag}")

    # estimated side (no inliers): drawn in white because black is invisible on this background
    z = np.linspace(Z_RANGE[0], Z_RANGE[1], 100)
    for key in ("ransac_sx", "ransac_dx"):
        l = out[key]
        if l is not None and l.get("inliers") is None:
            ax.plot(l["a"] * z + l["b"], z, "--", color="white", linewidth=2,
                    label=f"estimated ({l.get('source', 'default')})")

    ax.set(xlabel="x lateral (m)", ylabel="z forward (m)", title="Bird-Eye View: all RANSAC candidates (Canny)", xlim=X_RANGE, ylim=Z_RANGE)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()


def project_bev_xz(x, z, M_inv):
    """Road coordinates (x lateral, z forward, metres) -> pixel coordinates in the ORIGINAL image."""
    col = (np.asarray(x) - X_RANGE[0]) * PX_PER_M
    row = (Z_RANGE[1] - np.asarray(z)) * PX_PER_M
    pts = np.column_stack((col, row)).astype(np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, M_inv).reshape(-1, 2)


def plot_candidates_image(img, out, M_inv, out_path=None):
    """Original image + every RANSAC candidate (inliers + fitted line), same colours/legend as plot_candidates_thresh.
    Selected candidates: solid line. Unselected: dotted line."""
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(img)
    ax.axis("off")

    pts = out["pts"]
    role = {v: k for k, v in out["sel"].items() if v is not None}
    for k, c in enumerate(out["candidates"]):
        col = cmap(k % 10)
        inl = pts[c["inliers"]]
        p = project_bev_xz(inl[:, 0], inl[:, 1], M_inv)
        ax.scatter(p[:, 0], p[:, 1], s=4, color=col)
        z = np.linspace(inl[:, 1].min(), inl[:, 1].max(), 100)
        q = project_bev_xz(c["a"] * z + c["b"], z, M_inv)
        tag = f" -> {role[k].upper()}" if k in role else ""
        ax.plot(q[:, 0], q[:, 1], color=col, linewidth=2,
                linestyle="-" if k in role else ":",
                label=f"#{k} x@mid={x_at(c):+.2f} a={c['a']:+.2f} n={len(inl)}{tag}")

    # estimated side (no inliers): white dashed over a black underlay, to stay visible on bright asphalt
    z = np.linspace(Z_RANGE[0], Z_RANGE[1], 100)
    for key in ("ransac_sx", "ransac_dx"):
        l = out[key]
        if l is not None and l.get("inliers") is None:
            q = project_bev_xz(l["a"] * z + l["b"], z, M_inv)
            ax.plot(q[:, 0], q[:, 1], color="black", linewidth=4)
            ax.plot(q[:, 0], q[:, 1], "--", color="white", linewidth=2,
                    label=f"estimated ({l.get('source', 'default')})")

    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)            # imshow convention; keeps the image uncropped by the plotted lines
    ax.set_title("All RANSAC candidates on the original image")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()