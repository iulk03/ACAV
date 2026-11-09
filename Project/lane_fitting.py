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
MAX_SLOPE = 0.5             # pendenza massima (x = a*z + b), usata se non c'e' direzione di riferimento

# ---------------- direzione di riferimento dalla maschera ----------------
SLOPE_TOL = 0.08            # scarto massimo (in pendenza) dalla direzione della maschera (~4.6 gradi)
DIR_AGREE = 0.15            # se i due bordi della maschera differiscono di piu', uno dei due e' inaffidabile

# ---------------- voto a direzione fissa (al posto di RANSAC per le candidate) ----------------
USE_VOTING = True           # False -> torna al RANSAC con slope_ref (per confrontare le due versioni)
VOTE_BIN_M = 0.1            # larghezza dei bin dell'istogramma della posizione laterale "raddrizzata"
VOTE_MIN_ROWS = 20          # righe BEV distinte (a ~30 px/m: 20 righe ~ 0.7 m di linea) per avere una candidata
VOTE_NMS_M = 0.3            # due candidate non possono stare piu' vicine di cosi'
VOTE_BAND_M = 0.15          # semi-larghezza della banda di punti attorno al picco, usata per rifinire la retta
VOTE_REFINE_TOL = 0.03      # la pendenza rifinita non puo' scostarsi di piu' di cosi' da quella della maschera
VOTE_REFINE_SPAN_M = 4.0    # estensione minima in z dei punti per rifinire anche la pendenza (i tratteggi corti tengono quella della maschera)
VOTE_REFINE_BAND_M = 0.08   # per rifinire la pendenza si usano solo i punti entro questa distanza dalla retta preliminare
VOTE_WIDE_M = 0.35          # banda larga per misurare la rettilineita': una crepa sinuosa esce dalla banda stretta, una linea no

# ---------------- parabola (curve) ----------------
MIN_POLY_POINTS = 60            # min support points to fit a quadratic on a boundary
MIN_POLY_SPAN_FRAC = 0.5        # support points must span at least this fraction of the BEV depth
MAX_CURVATURE = 0.05            # 1/m: |kappa| above this (R < 20 m) is treated as a bad fit -> straight line

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
GUIDE_REJECT_M = 1.0         # a pair whose border is further than this from the (lane) guide border is rejected outright


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


def slope_ok(a, slope_ref=None, max_slope=MAX_SLOPE, tol=SLOPE_TOL):
    """Senza direzione di riferimento vale il limite assoluto, altrimenti la pendenza deve essere vicina a slope_ref."""
    if slope_ref is None:
        return abs(a) <= max_slope
    return abs(a - slope_ref) <= tol


def line_fit(p1, p2, separation=MIN_SEPARATION_M, max_slope=MAX_SLOPE, slope_ref=None):
    (x1, z1), (x2, z2) = p1, p2
    dz = z2 - z1
    if abs(dz) < separation:
        return None
    a = (x2 - x1) / dz
    if not slope_ok(a, slope_ref, max_slope):
        return None
    return a, x1 - a * z1


def ransac(points, threshold=INLINER_TRESHOLD_M, min_inliers=MIN_INLIERS, max_iter=MAX_ITER, p=SUCCESS_PROBABILITY,
           max_slope=MAX_SLOPE, sep_min=MIN_SEPARATION_M, seed=0, slope_ref=None):
    N = len(points)
    if N < max(2, min_inliers):
        return None
    rng = np.random.default_rng(seed)
    best, n_best, best_mask = None, 0, None
    k, n_iter, attempts = max_iter, 0, 0
    while n_iter < min(k, max_iter) and attempts < 50 * max_iter:
        attempts += 1
        i, j = rng.integers(0, N, size=2)
        line = line_fit(points[i], points[j], separation=sep_min, max_slope=max_slope, slope_ref=slope_ref)
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
    if not slope_ok(a, slope_ref, max_slope):
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


def car_inside(l, r):
    """Car (x=0) between the two lines, within CAR_TOL_M. Checked at z=0 AND at the nearest visible row (z=Z_RANGE[0]),
    and accepted if either says the car is inside: a straight extrapolation to z=0 is wrong on a curved road
    (um_000087: the line ends up at +0.8 m instead of -0.6 m), while z=7 is wrong on a straight but tilted one."""
    xl = min(l["b"], l["a"] * Z_RANGE[0] + l["b"])
    xr = max(r["b"], r["a"] * Z_RANGE[0] + r["b"])
    return xl < CAR_TOL_M and xr > -CAR_TOL_M


def near_weights(pts):
    """Per-point weight: `gain` at the closest z, linearly down to 1.0 at the farthest z."""
    t = (Z_RANGE[1] - pts[:, 1]) / (Z_RANGE[1] - Z_RANGE[0])      # 1 near the camera, 0 far
    return 1.0 + 2.0 * np.clip(t, 0.0, 1.0)


def find_candidates(points, n=N_CANDIDATES, threshold=INLINER_TRESHOLD_M, slope_ref=None):
    cands = []
    w = near_weights(points)
    remaining = np.arange(len(points))
    for _ in range(n):
        if len(remaining) < MIN_INLIERS:
            break
        sub = points[remaining]
        r = ransac(sub, slope_ref=slope_ref)
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


N_ROWS = int(round((Z_RANGE[1] - Z_RANGE[0]) * PX_PER_M))
_Z_ROW = Z_RANGE[1] - np.arange(N_ROWS) / PX_PER_M
W_ROW = near_weights(np.column_stack((np.zeros(N_ROWS), _Z_ROW)))      # peso di ogni riga BEV (vicino = piu' peso)


def vote_candidates(pts, a_ref, n=N_CANDIDATES):
    """Candidate lines with the direction FIXED to a_ref (slope of the mask).
    u = x - a_ref*z is the lateral position 'straightened' along the road direction: every line parallel to the road
    has a constant u. An histogram of u, counting DISTINCT BEV rows (not points) per bin, gives one peak per line.
    Dashed lines are short but cover many rows; cracks and shadow borders are sinuous, so they spread over many bins.
    """
    if len(pts) < VOTE_MIN_ROWS:
        return []
    u = pts[:, 0] - a_ref * pts[:, 1]
    rows = np.clip(np.round((Z_RANGE[1] - pts[:, 1]) * PX_PER_M).astype(int), 0, N_ROWS - 1)
    u0 = u.min()
    bins = ((u - u0) / VOTE_BIN_M).astype(int)
    nb = int(bins.max()) + 1

    occ = np.zeros((nb + 2, N_ROWS), bool)                 # occupied (bin, row) cells, padded by one bin per side
    occ[bins + 1, rows] = True
    occ3 = occ[:-2] | occ[1:-1] | occ[2:]                  # tolerance of +-1 bin (+-0.1 m)
    count = occ3.sum(1)                                    # distinct rows per bin
    score = (occ3 * W_ROW).sum(1)                          # same, weighted: near rows count more
    score[count < VOTE_MIN_ROWS] = 0.0

    k = int(round(VOTE_NMS_M / VOTE_BIN_M))
    cands = []
    for _ in range(n):
        b = int(np.argmax(score))
        if score[b] <= 0:
            break
        score[max(0, b - k):b + k + 1] = 0.0               # non-maximum suppression
        u_c = u0 + (b + 0.5) * VOTE_BIN_M
        near = np.abs(u - u_c) <= VOTE_BAND_M
        sel = pts[near]
        a, c = float(a_ref), float(np.median(sel[:, 0] - a_ref * sel[:, 1]))
        tight = sel[np.abs(sel[:, 0] - (a * sel[:, 1] + c)) <= VOTE_REFINE_BAND_M]
        if len(tight) >= MIN_INLIERS and np.ptp(tight[:, 1]) >= VOTE_REFINE_SPAN_M:
            a_fit, c_fit = np.polyfit(tight[:, 1], tight[:, 0], 1)
            if abs(a_fit - a_ref) <= VOTE_REFINE_TOL:
                a, c = float(a_fit), float(c_fit)
        d = line_distance(pts, a, c)
        inl = d <= INLINER_TRESHOLD_M
        if not inl.any():
            continue
        rows_in = np.unique(rows[inl])
        rows_wide = np.unique(rows[(d <= VOTE_WIDE_M) & (rows >= rows_in.min()) & (rows <= rows_in.max())])   # only along the same z-span
        straight = len(rows_in) / max(len(rows_wide), 1)       # ~1 for a straight line, <1 for a sinuous crack
        cands.append(dict(a=a, b=c, inliers=inl, n_iter=0, w_inl=float(W_ROW[rows_in].sum()) * straight ** 2))
    return cands


def pair_score(l, r, gl=None, gr=None):
    xl, xr = x_at(l), x_at(r)
    larg = xr - xl                                     # width: same at any z for ~parallel lines
    if not (LANE_W_MIN < larg < LANE_W_MAX):
        return -np.inf
    if abs(l["a"] - r["a"]) > SLOPE_DELTA_MAX:
        return -np.inf
    cl, cr = x_car(l), x_car(r)
    if not car_inside(l, r):
        return -np.inf
    if gl is not None and abs(xl - x_at(gl)) > GUIDE_REJECT_M:      # far from the ego-lane mask: wrong lane / spurious pair
        return -np.inf
    if gr is not None and abs(xr - x_at(gr)) > GUIDE_REJECT_M:
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
    """Fit x = a*z + b on the left/right border of a binary BEV mask (one point per row).
    max_slope is wide (1.0): the road direction is not known yet and the road can be diagonal in the BEV."""
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
    return ransac(np.array(pts), min_inliers=MASK_MIN_ROWS, max_slope=1.0)


def guide_borders(guide):
    """(left, right) border lines of the guide mask. Only if the guide is the ego-lane mask, else (None, None)."""
    if guide is None or guide.get("mask") is None or guide.get("source") != "lane":
        return None, None
    return mask_border_line(guide["mask"], "left"), mask_border_line(guide["mask"], "right")


def road_direction(guide):
    """Slope (x = a*z + b) of the guide mask, from its borders (road or ego-lane). None if not reliable.
    The mask is only used for the DIRECTION, not for the position: its borders can be off by ~0.5 m."""
    if guide is None or guide.get("mask") is None:
        return None
    ls = [mask_border_line(guide["mask"], s) for s in ("left", "right")]
    ls = [l for l in ls if l is not None]
    if not ls:
        return None
    if len(ls) == 2 and abs(ls[0]["a"] - ls[1]["a"]) > DIR_AGREE:
        ls = [max(ls, key=lambda l: l["inliers"].sum())]       # keep the better supported border
    w = [l["inliers"].sum() for l in ls]
    return float(np.average([l["a"] for l in ls], weights=w))


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


# ---------------- fit parabolico dopo il RANSAC ----------------
def support_points(line, pts):
    """Return the (x, z) points supporting a boundary, or None if they are not reliable (estimated line)."""
    if line.get("stimata"):
        return None
    if line.get("inliers") is not None:
        return pts[line["inliers"]]
    return None


def fit_boundary_poly(line, pts):
    straight = np.array([0.0, line["a"], line["b"]])
    p = support_points(line, pts)
    if p is None or len(p) < MIN_POLY_POINTS or np.ptp(p[:, 1]) < MIN_POLY_SPAN_FRAC * (Z_RANGE[1] - Z_RANGE[0]):
        return straight
    z, x = p[:, 1], p[:, 0]
    c, a, b = np.polyfit(z, x, 2)
    if abs(2 * c) > MAX_CURVATURE:
        return straight
    # the parabola must explain the points much better than the line, otherwise it is noise
    rss_lin = np.sum((x - np.polyval(np.polyfit(z, x, 1), z)) ** 2)
    rss_quad = np.sum((x - np.polyval([c, a, b], z)) ** 2)
    if rss_quad > 0.7 * rss_lin:
        return straight
    return np.array([c, a, b])


# ---------------- disegno e prospettiva ----------------
def line_on_image(line, M_inv, x_range=X_RANGE, z_range=Z_RANGE, px_per_m=PX_PER_M):
    z = np.linspace(z_range[0], z_range[1], 100)
    x = line.get("c", 0.0) * z ** 2 + line["a"] * z + line["b"]
    col = (x - x_range[0]) * px_per_m
    row = (z_range[1] - z) * px_per_m
    points_px = np.column_stack((col, row)).astype(np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points_px, M_inv).reshape(-1, 2)


def lane_polygon(line_lft, line_rgt, M_inv):
    """Polygon (original image pixels) between the two boundaries."""
    lf = line_on_image(line_lft, M_inv)
    rg = line_on_image(line_rgt, M_inv)
    return np.vstack((lf, rg[::-1])).astype(np.int32)


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
    a_ref = road_direction(guide)                       # direzione della maschera (strada o corsia ego)
    cands = vote_candidates(pts, a_ref) if (USE_VOTING and a_ref is not None) else []
    if not cands:                                       # nessuna direzione o nessun picco: RANSAC come prima
        cands = find_candidates(pts, slope_ref=a_ref)
    if not cands and a_ref is not None:                 # rete di sicurezza: direzione sbagliata -> senza vincolo
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

    out = dict(points=pts_dict, pts=pts, candidates=cands, ransac_sx=ransac_sx, ransac_dx=ransac_dx,
               stimata=stimata, sel=sel, guide=guide, slope_ref=a_ref)
    if ransac_sx and ransac_dx:
        curved = [dict(zip("cab", fit_boundary_poly(l, pts))) for l in (ransac_sx, ransac_dx)]
        out["overlay"] = draw_line(img, M_inv, *curved)
        pred = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(pred, [lane_polygon(*curved, M_inv)], 1)
        out["pred_mask"] = pred > 0                     # usata da main.py per l'IoU con la GT ego-lane
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

    # reference direction taken from the mask (magenta dotted, drawn through the car position)
    a_ref = out.get("slope_ref")
    if a_ref is not None:
        zz = np.linspace(Z_RANGE[0], Z_RANGE[1], 100)
        ax.plot(a_ref * (zz - Z_REF), zz, ":", color="magenta", linewidth=1.5, label=f"mask direction a={a_ref:+.2f}")

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