import matplotlib.pyplot as plt
import numpy as np
import cv2
from lane_pipeline import X_RANGE, Z_RANGE, PX_PER_M

# ---------------- parametri RANSAC (Ottimizzati per Canny) ----------------
MARGIN_PX = 6               # esclude i pixel troppo vicini ai bordi
INLINER_TRESHOLD_M = 0.06    # soglia in metri per essere inlier
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
VOTE_REFINE_TOL = 0.1      # la pendenza rifinita non puo' scostarsi di piu' di cosi' da quella della maschera
VOTE_REFINE_SPAN_M = 4.0    # estensione minima in z dei punti per rifinire anche la pendenza (i tratteggi corti tengono quella della maschera)
VOTE_REFINE_BAND_M = 0.08   # per rifinire la pendenza si usano solo i punti entro questa distanza dalla retta preliminare
VOTE_WIDE_M = 0.35          # banda larga per misurare la rettilineita': una crepa sinuosa esce dalla banda stretta, una linea no

# ---------------- parabola (curve) - dal secondo file ----------------
MIN_POLY_POINTS = 25         # min support points to fit a quadratic on a boundary
MIN_POLY_SPAN_FRAC = 0.35        # support points must span at least this fraction of the BEV depth
MAX_CURVATURE = 0.05            # 1/m: |kappa| above this (R < 20 m) is treated as a bad fit -> straight line
CURV_AGREE = 0.01               # max |c_left - c_right| (x = c*z^2 + ...); otherwise both boundaries stay straight
CURVE_BAND_M = 0.15             # band around the CURRENT model (line, then parabola) used to grow the support of a boundary along a curve
CURVE_GROW_ITER = 4             # growth iterations (each one widens the z window by CURVE_GROW_STEP_M per side)
CURVE_GROW_STEP_M = 2.0         # z window growth per side and per iteration (m)
LOOKAHEAD_M = 10.0              # distance from the vehicle of the red target point on the centreline
EGO_Z_M = 0.0                   # vehicle position along z
CAM_MERGE_Z_M = 12.0            # distanza z entro cui la traiettoria allineata alla camera converge alla mezzeria corsia
CAM_DRAW_START_Z_M = 4.5        # inizio disegno della traiettoria in avanti (dal bordo inferiore della camera)

# ---------------- parametri di scelta della coppia ----------------
N_CANDIDATES = 5            # quante rette candidate cercare
Z_REF = 0.5 * (Z_RANGE[0] + Z_RANGE[1])   # z di riferimento per la posizione laterale
LANE_W_MIN = 2.0            # larghezza minima plausibile della corsia (m)
LANE_W_MAX = 4.2            # max plausible lane width (m): wide intersections
SLOPE_DELTA_MAX = 0.1       # differenza massima di pendenza tra le due rette
W_CENTER = 55.0             # weak penalty for a lane not centred on the car
W_WIDTH = 120.0              # penalty per metre away from LANE_WIDTH_M
LANE_WIDTH_M = 3.2          # larghezza di default per il ripiego
MASK_EDGE_MARGIN_PX = 4     # border columns closer than this to the BEV side are fake (clipped by the ROI)
MASK_MIN_ROWS = 40          # min number of rows with a valid border point


CAR_TOL_M = 0.50            # car (x=0) must lie between the two lines, within this tolerance
W_SUPPORT_CAP = 220.0       # cap on the weighted inlier support of a single line
W_GUIDE = 80.0              # penalty per metre of disagreement with the guide border (lane guide only)
GUIDE_TOL_M = 0.6           # max distance for a candidate to be considered "matching" a guide border
GUIDE_REJECT_M = 1.0         # a pair whose border is further than this from the (lane) guide border is rejected outright


# ---------------- verifica corsia vs maschera strada (post-hoc) ----------------
ROAD_OK_FRAC = 0.75         # sopra questa frazione di poligono sulla strada: la scelta resta com'e'
ROAD_GOOD_FRAC = 0.85       # un'alternativa "buona" deve avere almeno questa frazione sulla strada...
ROAD_GAIN_GOOD = 0.10       # ...e migliorare di almeno questo rispetto alla scelta originale
ROAD_GAIN_ANY = 0.20        # se nessuna arriva a ROAD_GOOD_FRAC: la migliore passa solo con questo guadagno
ROAD_OK_FRAC_STRONG = 0.50  # soglia piu' bassa per una coppia di linee vere e ben supportate: non la si sostituisce con bordi maschera
STRONG_PAIR_SUPPORT = 100.0 # supporto pesato minimo di ciascuna delle due linee per considerare la coppia "forte"

# ---------------- guida ego troppo larga ----------------
EGO_GUIDE_MAX_W_M = 6.0     # se la maschera ego e' piu' larga (mediana per riga) non e' una corsia: la guida viene declassata a strada,
                            # i suoi bordi non vengono usati e la direzione si cerca con RANSAC senza vincolo (NON calibrato)


# ---------------- gate "ci sono marcature?" (solo vernice, lontano dai bordi strada) ----------------
INTERIOR_ERODE_M = 0.8      # la road_mask viene erosa di questo: bordo asfalto/erba, cordoli e ombre di bordo escono
MARKING_MIN_SUPPORT = 70.0  # supporto pesato minimo di una linea INTERNA alla strada; sotto -> niente marcature


# ---------------- integrazione obstacle detection ----------------
OBS_BORDER_TOL_PX = 2       # tolleranza (px) attorno a obs_body per ignorare i rientri della U-Net
W_OBS_SLICE = 250.0         # forte penalita' se un bordo corsia attraversa fisicamente un veicolo
W_OBS_PARKED = 120.0        # penalita' se una corsia troppo larga ingloba un veicolo laterale/parcheggiato
OBS_CENTER_TOL_M = 0.9      # veicoli con |x| < questa soglia sono considerati nella corsia ego (non penalizzati)

MAX_CAND_DIST_FROM_CAR = 3.5  # m — una striscia di corsia vera non può essere a più di 3.5 m dall'auto
SHARP_TURN_SLOPE = 0.6        # sopra questa soglia la direzione dalla maschera non è affidabile per la BEV
MAX_CENTER_OFFSET_M = 2.5     # la mezzeria corsia non può essere a più di 2.5 m dall'auto (sanity check traiettoria)


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
    z_near = Z_RANGE[0]
    xl_near = l["a"] * z_near + l["b"]
    xr_near = r["a"] * z_near + r["b"]
    xl_mid = 0.5 * (l["b"] + xl_near)
    xr_mid = 0.5 * (r["b"] + xr_near)
    return (xl_near < CAR_TOL_M and xr_near > -CAR_TOL_M) and (xl_mid < CAR_TOL_M and xr_mid > -CAR_TOL_M)


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


def obstacle_penalty(l, r, obs_boxes):
    """Penalizza coppie di bordi che attraversano fisicamente un ostacolo o inglobano veicoli laterali/parcheggiati."""
    if not obs_boxes:
        return 0.0
    pen = 0.0
    for box in obs_boxes:
        zs = np.array([box["z_min"], 0.5 * (box["z_min"] + box["z_max"]), box["z_max"]])
        xl = l["a"] * zs + l["b"]
        xr = r["a"] * zs + r["b"]
        x_obs_c = 0.5 * (box["x_min"] + box["x_max"])

        # 1) Un bordo della corsia attraversa fisicamente l'impronta del veicolo
        slices_l = np.any((box["x_min"] < xl) & (xl < box["x_max"]))
        slices_r = np.any((box["x_min"] < xr) & (xr < box["x_max"]))
        if slices_l or slices_r:
            pen += W_OBS_SLICE
        # 2) Veicolo laterale/parcheggiato interamente inglobato dentro una corsia troppo larga
        elif np.all((xl <= x_obs_c) & (x_obs_c <= xr)) and abs(x_obs_c) >= OBS_CENTER_TOL_M:
            pen += W_OBS_PARKED
    return pen


def pair_score(l, r, gl=None, gr=None, reject=GUIDE_REJECT_M, obs_boxes=()):
    xl, xr = x_at(l), x_at(r)
    larg = xr - xl                                     # width: same at any z for ~parallel lines
    guided = gl is not None and gr is not None
    if larg <= LANE_W_MIN or (larg >= LANE_W_MAX and not guided):
        return -np.inf
    if abs(l["a"] - r["a"]) > SLOPE_DELTA_MAX:
        return -np.inf
    cl, cr = x_car(l), x_car(r)
    if not car_inside(l, r):
        return -np.inf

    # Se la maschera ego copre tutta la carreggiata (> 4.2 m), consenti di dividerla
    # solo se la striscia interna ha supporto molto forte e sta ben dentro i bordi
    wide_guide = guided and (x_at(gr) - x_at(gl) > LANE_W_MAX)
    split_l = wide_guide and l["w_inl"] >= 130.0 and (x_at(gl) + 1.2 < xl < x_at(gr) - 1.2)
    split_r = wide_guide and r["w_inl"] >= 130.0 and (x_at(gl) + 1.2 < xr < x_at(gr) - 1.2)
    check_gl = gl is not None and not split_l
    check_gr = gr is not None and not split_r

    if check_gl and abs(xl - x_at(gl)) > reject:
        return -np.inf
    if check_gr and abs(xr - x_at(gr)) > reject:
        return -np.inf
    support = min(l["w_inl"], W_SUPPORT_CAP) + min(r["w_inl"], W_SUPPORT_CAP)
    penalty = W_CENTER * abs(0.5 * (cl + cr))
    if not guided or split_l or split_r:
        penalty += W_WIDTH * abs(larg - LANE_WIDTH_M)
    if check_gl:
        penalty += W_GUIDE * abs(xl - x_at(gl))
    if check_gr:
        penalty += W_GUIDE * abs(xr - x_at(gr))
    penalty += obstacle_penalty(l, r, obs_boxes)
    return support - penalty


def best_pair(cands, gl=None, gr=None, reject=GUIDE_REJECT_M, obs_boxes=()):
    best, best_s = None, -np.inf
    for l in cands:
        for r in cands:
            if l is r or x_at(l) >= x_at(r):
                continue
            s = pair_score(l, r, gl, gr, reject, obs_boxes=obs_boxes)
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
    strong = [c for c in cands if c["w_inl"] >= 0.25 * max(d["w_inl"] for d in cands)]   # ignore weak cracks
    return min(strong, key=lambda c: abs(x_at(c)))                                       # nearest to the car


def missing_side(line, width=LANE_WIDTH_M, flip=False):
    """Lato mancante a distanza `width`. Di default sul lato verso l'auto; flip=True -> sul lato opposto."""
    est = dict(a=line["a"], b=None, inliers=None, n_iter=0, stimata=True)
    if (x_at(line) > 0) != flip:
        est["b"] = line["b"] - width
        return est, line
    est["b"] = line["b"] + width
    return line, est


def mask_border_line(mask, side, px_per_m=PX_PER_M, x_range=X_RANGE, z_range=Z_RANGE, obs_body=None):
    """Fit x = a*z + b on the left/right border of a binary BEV mask (one point per row).
    max_slope is wide (1.0): the road direction is not known yet and the road can be diagonal in the BEV.
    Rows where the mask border touches `obs_body` are skipped so parked cars do not bend the border."""
    h, w = mask.shape
    pts = []
    for r in range(MARGIN_PX, h - MARGIN_PX):
        cols = np.flatnonzero(mask[r])
        if len(cols) == 0:
            continue
        c = cols[0] if side == "left" else cols[-1]
        if c <= MASK_EDGE_MARGIN_PX or c >= w - 1 - MASK_EDGE_MARGIN_PX:
            continue
        if obs_body is not None:
            c_lo = max(0, c - OBS_BORDER_TOL_PX)
            c_hi = min(w, c + OBS_BORDER_TOL_PX + 1)
            if obs_body[r, c_lo:c_hi].any():
                continue
        pts.append((x_range[0] + c / px_per_m, z_range[1] - r / px_per_m))
    if len(pts) < MASK_MIN_ROWS:
        return None
    return ransac(np.array(pts), min_inliers=MASK_MIN_ROWS, max_slope=1.0)


def guide_too_wide(guide):
    """True se la guida e' una maschera ego (source='lane') piu' larga di EGO_GUIDE_MAX_W_M (larghezza mediana per riga).
    Una maschera cosi' larga copre l'intera carreggiata (o sborda sul marciapiede): non e' una corsia, e i suoi bordi
    e la sua direzione porterebbero il fitting fuori strada."""
    if guide is None or guide.get("mask") is None or guide.get("source") != "lane":
        return False
    widths = (guide["mask"] > 0).sum(1) / PX_PER_M
    widths = widths[widths > 0]
    return len(widths) > 0 and float(np.median(widths)) > EGO_GUIDE_MAX_W_M


def guide_borders(guide):
    """(left, right) border lines of the guide mask. Only if the guide is the ego-lane mask, else (None, None)."""
    if guide is None or guide.get("mask") is None or guide.get("source") != "lane":
        return None, None
    obs_body = guide.get("obs_body")
    gl = mask_border_line(guide["mask"], "left", obs_body=obs_body)
    gr = mask_border_line(guide["mask"], "right", obs_body=obs_body)
    if gl is not None and gr is not None and x_at(gr) - x_at(gl) <= LANE_W_MIN:
        return None, None
    return gl, gr


def road_direction(guide):
    """Slope (x = a*z + b) of the guide mask, from its borders (road or ego-lane). None if not reliable.
    The mask is only used for the DIRECTION, not for the position: its borders can be off by ~0.5 m."""
    if guide is None or guide.get("mask") is None:
        return None
    obs_body = guide.get("obs_body")
    ls = [mask_border_line(guide["mask"], s, obs_body=obs_body) for s in ("left", "right")]
    ls = [l for l in ls if l is not None]
    if not ls:
        return None
    if len(ls) == 2 and abs(ls[0]["a"] - ls[1]["a"]) > DIR_AGREE:
        if guide.get("source") == "road":
            return None
        ls = [max(ls, key=lambda l: l["inliers"].sum())]
    w = [l["inliers"].sum() for l in ls]
    return float(np.average([l["a"] for l in ls], weights=w))


def missing_side_from_guide(line, guide):
    """Other lane border taken from the mask edge. Returns (left, right) or None."""
    if guide is None or guide.get("mask") is None:
        return None
    known_is_right = x_at(line) > 0
    obs_body = guide.get("obs_body")
    edge = mask_border_line(guide["mask"], "left" if known_is_right else "right", obs_body=obs_body)
    if edge is None:
        return None
    width = x_at(line) - x_at(edge) if known_is_right else x_at(edge) - x_at(line)
    if not (LANE_W_MIN < width < LANE_W_MAX) or abs(edge["a"] - line["a"]) > SLOPE_DELTA_MAX:
        return None
    est = dict(a=edge["a"], b=edge["b"], inliers=None, n_iter=0, stimata=True, source=guide["source"])
    return (est, line) if known_is_right else (line, est)


# ---------------- frazione di corsia sulla strada + alternative ----------------
def _straight(x):
    return dict(c=0.0, a=x["a"], b=x["b"], zr=Z_RANGE)


def _borrow_shape(ref, other):
    """Boundary `other` (estimated or too short to give a curvature) rebuilt as a copy of the curved boundary `ref`,
    shifted laterally so that it keeps its own position at Z_REF. The two borders of a lane share the curvature."""
    d = x_at(other) - (ref["c"] * Z_REF ** 2 + ref["a"] * Z_REF + ref["b"])
    return dict(c=ref["c"], a=ref["a"], b=ref["b"] + d, zr=ref["zr"])


def curved_pair(l, r, pts):
    """Fit parabolico dei due bordi (logica diretta presa dal secondo file, con regola di curvatura condivisa)."""
    raw = (l, r)
    curved = [fit_boundary_poly(x, pts) for x in raw]
    for i in (0, 1):
        if raw[i].get("stimata") and not raw[1 - i].get("stimata"):
            other = curved[1 - i]
            curved[i] = dict(
                c=other["c"],
                a=other["a"] + (raw[i]["a"] - raw[1 - i]["a"]),
                b=other["b"] + (raw[i]["b"] - raw[1 - i]["b"]),
                zr=other["zr"],
            )
    if abs(curved[0]["c"] - curved[1]["c"]) > CURV_AGREE:     # curvature must be shared by both borders
        curved = [dict(c=0.0, a=x["a"], b=x["b"], zr=Z_RANGE) for x in raw]
    return curved


def boundary_x(line, z=_Z_ROW):
    """x del bordo per ogni riga BEV (stessa formula di line_on_image: parabola nel supporto, tangente fuori)."""
    zc = np.clip(z, *line["zr"])
    c = line["c"]
    return c * zc ** 2 + line["a"] * zc + line["b"] + (2 * c * zc + line["a"]) * (z - zc)


def lane_bev_mask(curved, shape):
    """Poligono corsia nel BEV (bool), una riga alla volta tra bordo sinistro e destro."""
    w = shape[1]
    xl, xr = (boundary_x(l) for l in curved)
    cl, cr = (xl - X_RANGE[0]) * PX_PER_M, (xr - X_RANGE[0]) * PX_PER_M
    cols = np.arange(w)[None, :] + 0.5
    return (cols >= cl[:, None]) & (cols <= cr[:, None])


def lane_road_frac(l, r, pts, road, valid):
    """Frazione del poligono corsia (nella parte visibile del BEV) che sta sulla road_mask."""
    lane = lane_bev_mask(curved_pair(l, r, pts), road.shape) & valid
    area = int(lane.sum())
    if area == 0:
        return 0.0
    return float((lane & (road > 0)).sum() / area)


def lane_from_road(road, width=LANE_WIDTH_M, obs_body=None, full_road=True):
    gl = mask_border_line(road, "left", obs_body=obs_body)
    gr = mask_border_line(road, "right", obs_body=obs_body)
    mk = lambda l: dict(a=l["a"], b=l["b"], inliers=None, n_iter=0, stimata=True, source="road")
    if gl is not None and gr is not None:
        road_w = x_at(gr) - x_at(gl)
        if full_road or road_w <= LANE_W_MAX:
            return mk(gl), mk(gr)
        # Strada più larga di una corsia con marcature interne: scegliere il bordo più vicino all'auto (x=0)
        if abs(x_at(gl)) < abs(x_at(gr)):
            # Bordo sinistro più vicino → usa quello come riferimento
            left = mk(gl)
            right = mk(gl)
            right["b"] = gl["b"] + width
            return left, right
        else:
            # Bordo destro più vicino → usa quello come riferimento
            left = mk(gr)
            left["b"] = gr["b"] - width
            return left, mk(gr)
    if gr is not None:
        right = mk(gr)
        left = mk(gr)
        left["b"] = gr["b"] - width
        return left, right
    if gl is not None:
        left = mk(gl)
        right = mk(gl)
        right["b"] = gl["b"] + width
        return left, right
    return None


def road_alternatives(cands, pair, gl, gr, road, obs_body=None, obs_boxes=(), no_marks=True):
    """Alternative alla scelta originale, nell'ordine: (1) altre coppie valide, (2) una candidata con il lato
    mancante da bordo maschera / default / lato opposto, (3) corsia dalla sola road_mask.
    Ogni elemento: (nome, sinistra, destra, stimata, sel)."""
    alts = []

    # 1) altre coppie valide (con o senza la guida), dalla migliore
    pairs = []
    for i, l in enumerate(cands):
        for j, r in enumerate(cands):
            if i == j or x_at(l) >= x_at(r):
                continue
            if pair is not None and l is pair[0] and r is pair[1]:
                continue
            s = max(
                pair_score(l, r, gl, gr, obs_boxes=obs_boxes),
                pair_score(l, r, obs_boxes=obs_boxes),
            )
            if np.isfinite(s):
                pairs.append((s, i, j))
    for s, i, j in sorted(pairs, reverse=True):
        alts.append(("pair", cands[i], cands[j], False, dict(left=i, right=j)))

    # 2) candidata singola, lato mancante in tre modi
    road_guide = dict(mask=road, source="road", obs_body=obs_body)
    for i in sorted(range(len(cands)), key=lambda k: -cands[k]["w_inl"]):
        c = cands[i]
        side = "right" if x_at(c) > 0 else "left"
        sel = dict(left=None, right=None)
        sel[side] = i
        for name, res in (("single+mask", missing_side_from_guide(c, road_guide)),
                          ("single+default", missing_side(c)),
                          ("single+flip", missing_side(c, flip=True))):
            if res is not None:
                alts.append((name, res[0], res[1], True, dict(sel)))

    # 3) solo road_mask
    res = lane_from_road(road, obs_body=obs_body, full_road=no_marks)
    if res is not None:
        alts.append(("road-only", res[0], res[1], True, dict(left=None, right=None)))
    return alts


def pick_alternative(alts, f0, pts, road, valid, obs_boxes=()):
    """Prima alternativa (in ordine) con frac >= ROAD_GOOD_FRAC e guadagno >= ROAD_GAIN_GOOD.
    Se non c'e': la migliore in assoluto, solo se il guadagno e' >= ROAD_GAIN_ANY. Altrimenti None.
    Penalizza leggermente le alternative che attraversano o inglobano veicoli laterali."""
    scored = []
    for a in alts:
        f = lane_road_frac(a[1], a[2], pts, road, valid)
        obs_pen = obstacle_penalty(a[1], a[2], obs_boxes)
        f_eff = f - 0.25 * (obs_pen / W_OBS_SLICE)
        scored.append((f, f_eff, a))

    for f, f_eff, a in scored:
        if f >= ROAD_GOOD_FRAC and f_eff - f0 >= ROAD_GAIN_GOOD:
            return a, f
    if scored:
        f, f_eff, a = max(scored, key=lambda t: t[1])     # a parita' vince la prima in ordine
        if f_eff - f0 >= ROAD_GAIN_ANY:
            return a, f
    return None


def _index(cands, c):
    return next((i for i, x in enumerate(cands) if x is c), None)   # identity: dicts hold numpy arrays


# ---------------- fit parabolico (preso dal secondo file) ----------------
def support_points(line, pts):
    """Return the (x, z) points supporting a boundary, or None if they are not reliable (estimated line)."""
    if line.get("stimata"):
        return None
    if line.get("inliers") is not None:
        return pts[line["inliers"]]
    return None


def grow_support(p0, pts):
    """Support of a boundary that follows the CURVE instead of the straight line.
    The straight inliers (band 0.06 m) only cover the tangent segment of a curved line. Starting from them, the model
    (a line first, a parabola as soon as the support is long enough) is refitted and the support is re-collected in a
    band of CURVE_BAND_M around the model, widening the z window a little each time. Straight lines stay as they are
    (the parabola is then rejected by the RSS test in fit_boundary_poly)."""
    depth = Z_RANGE[1] - Z_RANGE[0]
    zlo, zhi = float(p0[:, 1].min()), float(p0[:, 1].max())
    a, b = np.polyfit(p0[:, 1], p0[:, 0], 1)
    coef = np.array([0.0, a, b])
    p = p0
    for _ in range(CURVE_GROW_ITER):
        zlo, zhi = max(Z_RANGE[0], zlo - CURVE_GROW_STEP_M), min(Z_RANGE[1], zhi + CURVE_GROW_STEP_M)
        near = (pts[:, 1] >= zlo) & (pts[:, 1] <= zhi) & (np.abs(pts[:, 0] - np.polyval(coef, pts[:, 1])) <= CURVE_BAND_M)
        q = pts[near]
        if len(q) < MIN_POLY_POINTS:
            break
        if np.ptp(q[:, 1]) >= MIN_POLY_SPAN_FRAC * depth:
            new = np.polyfit(q[:, 1], q[:, 0], 2)
        else:
            new = np.array([0.0, *np.polyfit(q[:, 1], q[:, 0], 1)])
        if abs(2 * new[0]) > MAX_CURVATURE:              # implausible curvature: stop growing, keep the last good support
            break
        converged = len(q) == len(p) and np.allclose(new, coef, atol=1e-4)
        p, coef = q, new
        if converged:
            break
    return p


def fit_boundary_poly(line, pts):
    """Quadratic x = c*z^2 + a*z + b of one boundary. 'kind' tells why the result is what it is:
      curved   -> parabola accepted
      straight -> enough support, but the data do not justify a curvature (or it is implausible)
      weak     -> estimated border / not enough support: no information about the curvature"""
    straight = dict(c=0.0, a=line["a"], b=line["b"], zr=Z_RANGE, kind="weak")
    p0 = support_points(line, pts)
    if p0 is None or len(p0) < MIN_POLY_POINTS:
        return straight
    p = grow_support(p0, pts)
    if len(p) < MIN_POLY_POINTS or np.ptp(p[:, 1]) < MIN_POLY_SPAN_FRAC * (Z_RANGE[1] - Z_RANGE[0]):
        return straight
    straight["kind"] = "straight"
    z, x = p[:, 1], p[:, 0]
    c, a, b = np.polyfit(z, x, 2)
    if abs(2 * c) > MAX_CURVATURE:
        return straight
    # the parabola must explain the points much better than a line, otherwise it is noise
    rss_lin = np.sum((x - np.polyval(np.polyfit(z, x, 1), z)) ** 2)
    rss_quad = np.sum((x - np.polyval([c, a, b], z)) ** 2)
    if rss_quad > 0.7 * rss_lin:
        return straight
    return dict(c=float(c), a=float(a), b=float(b), zr=(float(z.min()), float(z.max())), kind="curved", n=int(len(p)))


# ---------------- centreline (che parte dalla camera), curvature, lookahead ----------------
def ego_centerline(curved, z_merge=CAM_MERGE_Z_M):
    """Traiettoria centrale che parte dalla camera in (x=0, z=0, dx/dz=0) e si raccorda
    dolcemente alla mezzeria geometrica della corsia entro `z_merge`."""
    z = np.linspace(Z_RANGE[0], Z_RANGE[1], 100)
    x = 0.5 * (boundary_x(curved[0], z) + boundary_x(curved[1], z))
    c_l, a_l, b_l = [float(v) for v in np.polyfit(z, x, 2)]

    # Posizione e pendenza della mezzeria corsia al punto di raccordo z_merge
    zm = float(z_merge)
    xm = c_l * zm ** 2 + a_l * zm + b_l
    sm = 2.0 * c_l * zm + a_l

    # Cubica di Hermite x(z) = d*z^3 + c*z^2 con x(0)=0, x'(0)=0, x(zm)=xm, x'(zm)=sm
    c_h = (3.0 * xm - sm * zm) / (zm ** 2)
    d_h = (sm * zm - 2.0 * xm) / (zm ** 3)
    return dict(
        d=float(d_h),
        c=float(c_h),
        a=0.0,
        b=0.0,
        zm=zm,
        lane_c=c_l,
        lane_a=a_l,
        lane_b=b_l,
    )


def eval_centerline(center, z):
    """Valuta la traiettoria che parte dalla camera per un array o scalare z."""
    z_arr = np.asarray(z, dtype=float)
    zm = center.get("zm")
    if zm is None:
        return center["c"] * z_arr ** 2 + center["a"] * z_arr + center["b"]
    d = center.get("d", 0.0)
    x_hermite = d * z_arr ** 3 + center["c"] * z_arr ** 2 + center["a"] * z_arr + center["b"]
    x_lane = center["lane_c"] * z_arr ** 2 + center["lane_a"] * z_arr + center["lane_b"]
    return np.where(z_arr <= zm, x_hermite, x_lane)


def curvature_at(center, z):
    """Signed curvature [1/m] of the trajectory at depth z (positive = curving right) and radius [m]."""
    zm = center.get("zm")
    if zm is not None and z <= zm:
        d = center.get("d", 0.0)
        slope = 3.0 * d * z ** 2 + 2.0 * center["c"] * z + center["a"]
        d2x = 6.0 * d * z + 2.0 * center["c"]
    elif zm is not None:
        slope = 2.0 * center["lane_c"] * z + center["lane_a"]
        d2x = 2.0 * center["lane_c"]
    else:
        slope = 2.0 * center["c"] * z + center["a"]
        d2x = 2.0 * center["c"]
    kappa = d2x / (1.0 + slope ** 2) ** 1.5
    radius = 1.0 / abs(kappa) if abs(kappa) > 1e-6 else 1e6
    return float(kappa), float(min(radius, 1e6))


def lookahead_point(center, lookahead=LOOKAHEAD_M):
    """First point of the camera-aligned centreline at distance >= lookahead from the vehicle (x=0, z=EGO_Z_M)."""
    z = np.linspace(CAM_DRAW_START_Z_M, Z_RANGE[1], 250)
    x = eval_centerline(center, z)
    ok = np.flatnonzero(np.hypot(x, z - EGO_Z_M) >= lookahead)
    i = int(ok[0]) if len(ok) else len(z) - 1
    return float(x[i]), float(z[i])


def draw_prediction(img, M_inv, center, target):
    """Blue centreline starting from the bottom edge of the camera and red lookahead dot on the image (in place)."""
    z = np.linspace(CAM_DRAW_START_Z_M, Z_RANGE[1], 300)
    x = eval_centerline(center, z)
    pts_uv = project_bev_xz(x, z, M_inv)
    h, w = img.shape[:2]
    inside = (pts_uv[:, 0] >= 0) & (pts_uv[:, 0] < w) & (pts_uv[:, 1] >= 0) & (pts_uv[:, 1] < h)
    if inside.sum() >= 2:
        cv2.polylines(img, [pts_uv[inside].astype(np.int32)], False, (0, 0, 255), 3, cv2.LINE_AA)
    u, v = project_bev_xz([target[0]], [target[1]], M_inv)[0]
    if 0 <= u < w and 0 <= v < h:
        cv2.circle(img, (int(u), int(v)), 8, (255, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (int(u), int(v)), 10, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------- disegno e prospettiva ----------------
def line_on_image(line, M_inv, x_range=X_RANGE, z_range=Z_RANGE, px_per_m=PX_PER_M):
    z = np.linspace(z_range[0], z_range[1], 100)
    zc = np.clip(z, *line.get("zr", z_range))          # parabola only inside the support
    c = line.get("c", 0.0)
    x = c * zc ** 2 + line["a"] * zc + line["b"] + (2 * c * zc + line["a"]) * (z - zc)   # tangent outside
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


def marking_support(img_tresh, road, a_ref, obs_ground=None):
    """Supporto pesato della migliore linea candidata che sta DENTRO la strada (road_mask erosa di INTERIOR_ERODE_M).
    I bordi della carreggiata restano fuori dalla maschera erosa: quello che rimane e' quasi solo vernice.
    Esclude anche l'impronta a terra degli ostacoli rilevati."""
    k = int(round(INTERIOR_ERODE_M * PX_PER_M))
    inner = cv2.erode((road > 0).astype(np.uint8) * 255, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    if obs_ground is not None and obs_ground.any():
        inner = cv2.bitwise_and(inner, cv2.bitwise_not(obs_ground))
    d = points_in_meters(cv2.bitwise_and(img_tresh, inner))
    p = np.column_stack((d["x"], d["z"]))
    if len(p) < MIN_INLIERS:
        return 0.0
    cands = vote_candidates(p, a_ref) if a_ref is not None else []
    if not cands:
        cands = find_candidates(p, slope_ref=a_ref)
    return max((c["w_inl"] for c in cands), default=0.0)


def lane_from_masks(gl, gr, road, obs_body=None):
    """Stima senza marcature: bordi della maschera ego se ci sono entrambi, altrimenti corsia dalla sola road_mask.
    Restituisce (sinistra, destra, nome) oppure None."""
    if gl is not None and gr is not None:
        mk = lambda l: dict(a=l["a"], b=l["b"], inliers=None, n_iter=0, stimata=True, source="lane")
        return mk(gl), mk(gr), "no-marks:ego"
    res = lane_from_road(road, obs_body=obs_body, full_road=True)
    if res is not None:
        return res[0], res[1], "no-marks:road"
    return None


# ---------------- funzione principale di fitting ----------------
def lane_fitting(img_tresh, img, M_inv, guide=None):

    pts_dict = points_in_meters(img_tresh)
    pts = np.column_stack((pts_dict["x"], pts_dict["z"]))

    ego_wide = guide_too_wide(guide)
    if ego_wide:                                        # maschera ego troppo larga: declassata a guida "road", niente bordi ne' direzione
        road_m = guide.get("road")
        guide = {**guide, "source": "road", "mask": road_m if (road_m is not None and road_m.any()) else None}
    gl, gr = guide_borders(guide)
    a_ref = None if ego_wide else road_direction(guide)  # direzione della maschera (strada o corsia ego)
    # Curva troppo stretta: la BEV distorce troppo, la direzione dalla maschera non è affidabile
    if a_ref is not None and abs(a_ref) > SHARP_TURN_SLOPE:
        a_ref = None
    road = guide.get("road") if guide is not None else None
    valid = guide.get("valid") if guide is not None else None
    obs_body = guide.get("obs_body") if guide is not None else None
    obs_ground = guide.get("obs_ground") if guide is not None else None
    obs_boxes = guide.get("obs_boxes", ()) if guide is not None else ()
    have_road = road is not None and valid is not None and road.any()

    cands = vote_candidates(pts, a_ref) if (USE_VOTING and a_ref is not None) else []
    if not cands:                                       # nessuna direzione o nessun picco: RANSAC come prima
        cands = find_candidates(pts, slope_ref=a_ref)
    if not cands and a_ref is not None:                 # rete di sicurezza: direzione sbagliata -> senza vincolo
        cands = find_candidates(pts)

    # Filtrare candidate troppo lontane dal veicolo quando la guida ego è affidabile
    if guide is not None and guide.get("source") == "lane" and not ego_wide and cands:
        cands = [c for c in cands if abs(x_at(c)) <= MAX_CAND_DIST_FROM_CAR]

    # ---- gate: ci sono marcature dentro la strada? Se no, si usa un cordolo forte o le maschere ----
    msup = marking_support(img_tresh, road, a_ref, obs_ground=obs_ground) if have_road else None
    min_mark_thr = MARKING_MIN_SUPPORT
    if have_road and obs_ground is not None and obs_ground.any():
        visible_frac = float(((road > 0) & (obs_ground == 0)).sum() / (road > 0).sum())
        min_mark_thr = MARKING_MIN_SUPPORT * float(np.clip(visible_frac, 0.4, 1.0))
    no_marks = msup is not None and msup < min_mark_thr
    if no_marks:
        for c in cands:
            c["w_inl"] *= 0.5   # declassa ma non elimina: evita di buttare tutto e ricadere su maschere imprecise
        cands = [c for c in cands if c["w_inl"] >= min_mark_thr * 0.5]
    fallback = lane_from_masks(gl, gr, road, obs_body=obs_body) if (no_marks and not cands) else None

    ransac_sx = ransac_dx = None
    stimata = False
    sel = dict(left=None, right=None)
    road_frac, alt_name = None, None

    if fallback is not None:
        ransac_sx, ransac_dx, alt_name = fallback
        stimata = True
        if have_road:
            road_frac = lane_road_frac(ransac_sx, ransac_dx, pts, road, valid)
    else:
        pair = best_pair(cands, gl, gr, obs_boxes=obs_boxes)
        hits = [any(abs(x_at(c) - x_at(g)) < GUIDE_TOL_M for c in cands) for g in (gl, gr) if g is not None]
        odd_width = gl is not None and gr is not None and abs(x_at(gr) - x_at(gl) - LANE_WIDTH_M) > GUIDE_REJECT_M
        trusted = all(hits) if odd_width else any(hits)
        if pair is None and not trusted:    # guide unreliable: drop it (an odd-width mask needs both borders confirmed)
            pair = best_pair(cands, obs_boxes=obs_boxes)

        if pair is not None:
            ransac_sx, ransac_dx = pair
            sel = dict(left=_index(cands, pair[0]), right=_index(cands, pair[1]))
        elif cands:
            best = pick_anchor(cands, gl, gr)
            result = missing_side_from_guide(best, guide)
            if result is None and gl is not None and gr is not None:
                # Usa il bordo della maschera ego dal lato mancante
                known_is_right = x_at(best) > 0
                edge = gl if known_is_right else gr
                width_check = abs(x_at(best) - x_at(edge))
                if LANE_W_MIN < width_check < LANE_W_MAX:
                    est = dict(a=edge["a"], b=edge["b"], inliers=None, n_iter=0, stimata=True, source="lane")
                    result = (est, best) if known_is_right else (best, est)
            if result is None:
                result = missing_side(best)
            ransac_sx, ransac_dx = result
            sel["right" if x_at(best) > 0 else "left"] = _index(cands, best)
            stimata = True

        # ---- verifica sulla road_mask: se la corsia scelta sta (quasi) tutta sulla strada non cambia nulla ----
        if have_road:
            road_frac = lane_road_frac(ransac_sx, ransac_dx, pts, road, valid) if (ransac_sx and ransac_dx) else 0.0
            # coppia di linee vere (non stimate) e ben supportate: la si sostituisce solo se la road_mask la contraddice nettamente
            strong_pair = (pair is not None and ransac_sx is pair[0] and ransac_dx is pair[1]
                           and min(pair[0]["w_inl"], pair[1]["w_inl"]) >= STRONG_PAIR_SUPPORT)
            ok_frac = ROAD_OK_FRAC_STRONG if strong_pair else ROAD_OK_FRAC
            if road_frac < ok_frac:
                alts = road_alternatives(cands, pair, gl, gr, road, obs_body=obs_body, obs_boxes=obs_boxes, no_marks=no_marks)
                choice = pick_alternative(alts, road_frac, pts, road, valid, obs_boxes=obs_boxes)
                if choice is not None:
                    (alt_name, ransac_sx, ransac_dx, stimata, sel), road_frac = choice

    out = dict(points=pts_dict, pts=pts, candidates=cands, ransac_sx=ransac_sx, ransac_dx=ransac_dx,
               stimata=stimata, sel=sel, guide=guide, slope_ref=a_ref, road_frac=road_frac, alt=alt_name,
               marking_support=msup, no_marks=no_marks, ego_too_wide=ego_wide,
               supports=sorted((c["w_inl"] for c in cands), reverse=True))   # diagnostic: weighted support of every candidate
    if ransac_sx and ransac_dx:
        curved = curved_pair(ransac_sx, ransac_dx, pts)
        out["overlay"] = draw_line(img, M_inv, *curved)
        pred = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(pred, [lane_polygon(*curved, M_inv)], 1)
        out["pred_mask"] = pred > 0                     # usata da main.py per l'IoU con la GT ego-lane
        out["curved"] = curved
        center = ego_centerline(curved)
        kappa, radius = curvature_at(center, Z_RANGE[0])
        target = lookahead_point(center)
        out.update(center=center, curvature=kappa, radius=radius, target=target)
        # Sanity check: se la mezzeria corsia è troppo lontana dall'auto la predizione non è affidabile
        center_x_at_ref = 0.5 * (boundary_x(curved[0], np.array([Z_REF]))[0]
                                + boundary_x(curved[1], np.array([Z_REF]))[0])
        if abs(center_x_at_ref) <= MAX_CENTER_OFFSET_M:
            draw_prediction(out["overlay"], M_inv, center, target)
        else:
            out["center_rejected"] = True
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
    for l in fitting_out.get("curved", []):
        z = np.linspace(*l["zr"], 100)
        ax.plot(l["c"] * z ** 2 + l["a"] * z + l["b"], z, "w--", linewidth=1)

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
    if guide is not None and guide.get("obs_ground") is not None and guide["obs_ground"].any():
        ax.contour((guide["obs_ground"] > 0).astype(float), levels=[0.5], colors="orange", linewidths=1.5, extent=(X_RANGE[0], X_RANGE[1], Z_RANGE[0], Z_RANGE[1]), origin="upper")
        ax.plot([], [], color="orange", label="obstacle footprint")

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