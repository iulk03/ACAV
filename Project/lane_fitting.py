import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import cv2
from lane_pipeline import X_RANGE, Z_RANGE, PX_PER_M

# ---------------- RANSAC parameters ----------------
MARGIN_PX = 6                   # ignore pixels too close to the image border
INLINER_TRESHOLD_M = 0.2        # max distance to be an inlier
MIN_SEPARATION_M = 2.0          # min z distance between the two points defining a line
SUCCESS_PROBABILITY = 0.99
MAX_ITER = 500
MIN_INLIERS = 100
MAX_SLOPE = 0.4                 # max slope of a single line (x = a*z + b)
SLOPE_TOL = 0.06                # painted lines: |a - a_ref| <= SLOPE_TOL, a_ref from the road edges
COV_BIN_M = 0.1                 # z-bin used to measure how much of the line is actually covered
BOTTOM_MARGIN_PX = 1            # near edge: keep almost every row (MARGIN_PX still applies to top/left/right)
CURVE_MIN_GAIN = 1.1            # the quadratic model must cover >=10% more depth than the straight one to be accepted
CURVE_SLOPE_TOL = 0.15          # max |dx/dz - a_ref| allowed at BOTH ends of the BEV for a quadratic hypothesis
RANSAC_MAX_CURVATURE = 0.02     # 1/m: |2c| above this (R < 50 m) is rejected as a RANSAC hypothesis
TIGHT_THRESHOLD_M = 0.08        # half-width of the "painted stroke" band around a line
TIGHT_WEIGHT = 1.0              # RANSAC score = loose coverage + TIGHT_WEIGHT * tight coverage
TIGHT_BONUS = 2.0               # line_score bonus per metre of line covered by tight inliers

# ---------------- candidate and lane parameters ----------------
SNAP_MAX_DIST_M = 0.3           # a painted line replaces the mask edge only if it is this close to it
SNAP_MIN_COV_FRAC = 0.5         # and covers at least this fraction of the BEV depth
INTERIOR_MIN_SCORE_FRAC = 0.4   # interior line score vs the best interior line
EDGE_BAND_M = 0.3               # RANSAC points are searched between the edges, widened by this band
NARROW_MIN_COV_FRAC = 0.4       # a line closer than MIN_LANE_WIDTH_M to a road edge must cover at least this fraction of the BEV depth

N_CANDIDATES = 15               # how many candidate lines to search for
SLOPE_DELTA_MAX = 0.4           # max slope difference to be considered parallel
MERGE_DIST_M = 0.5              # painted line vs road edge: closer than this = duplicate of the edge marking
MIN_LANE_WIDTH_M = 2.5          # painted vs painted: no real lane is narrower than this
MAX_LANE_WIDTH_M = 5.0          # a lane wider than this (unless bounded by two road edges) is not a real lane
ROAD_EDGE_THRESHOLD_M = 0.25
ROAD_EDGE_MIN_INLIERS = 40
ROAD_GAP_M = 3.5                # max non-road gap between the vehicle column and the first road run (shadow holes)
EDGE_PARALLEL_TOL = 0.12
EGO_Z_M = 0.0
EGO_CHECK_Z_M = Z_RANGE[0]      # depth at which the ego lane is decided (no extrapolation outside the BEV)
PAINT_MIN_FRAC = 0.3            # start at 0.0 (filter off): first measure, then set it
SNAP_MIN_RIDGE_FRAC = 0.4       # snap: ridge coverage (fraction of BEV depth) accepted as paint evidence
EGO_AXIS_TOL_M = 0.8            # a lane is an ego candidate if the vehicle axis falls inside it, widened by this tolerance
EDGE_SNAP_RIDGE_FRAC = 0.15     # edge clusters: the road mask already gives a prior, so less paint evidence is enough (~2 m)
EGO_EDGE_HALF_W_M = 0.5         # half-width of the strips left/right of a line where ego_prob is compared
EGO_EDGE_MIN_CONTRAST = 0.3     # min |mean ego_prob right - left| to trust the network over line_score

RIDGE_REF_PCT = 99.9            # reference ridge strength: this percentile of the ridge image
RIDGE_STRONG_FRAC = 0.3         # a z-bin counts as "on paint" if the ridge near the line exceeds this fraction of the reference
RIDGE_HALF_W_M = 0.15           # half-width of the window around the line where the ridge is searched
RIDGE_BONUS = 2.0               # score bonus per metre of line lying on a strong ridge

COLOR_BONUS = 2.0               # score bonus per metre of line lying on the colour mask
EGO_EDGE_HALF_W_M = 0.5         # half-width of the strips left/right of a line where ego_prob is compared
EGO_EDGE_MIN_CONTRAST = 0.3     # min |mean ego_prob right - left| for a z-bin to count as "on the ego-lane edge"
EGO_EDGE_BONUS = 3.0            # score bonus per metre of line lying on an ego-lane edge (additive only)
EGO_PROB_MIN_PEAK = 0.5         # if the network never exceeds this, it sees no ego lane: no bonus at all

# ---------------- curvature and prediction (ego lane) ----------------
MIN_POLY_POINTS = 60            # min support points to fit a quadratic on a boundary
MIN_POLY_SPAN_FRAC = 0.5        # support points must span at least this fraction of the BEV depth
POLY_MAX_CURVATURE = 0.01       # 1/m: |2c| above this (R < 100 m) -> keep the straight line
POLY_MIN_GAIN = 0.7             # quadratic rms must be below this fraction of the straight rms
POLY_STRAIGHT_RMS_M = 0.05      # if the straight fit is already this good (5 cm), do not curve
MAX_CURVATURE = 0.05            # 1/m: |kappa| above this (R < 20 m) is treated as a bad fit -> straight line
LOOKAHEAD_M = 10.0              # distance from the vehicle (z = EGO_Z_M) of the red target point
D_MIN_M = 5.0                   # minimum gap to the lead vehicle
T_GAP_S = 1.5                   # time gap for v_max
FOOT_SHRINK = 0.6               # use only the central 60% of the bbox width for the lane-overlap test

CAND_COLORS = ["red", "cyan", "lime", "magenta", "orange", "dodgerblue", "gold", "deeppink", "white", "mediumpurple"]
EGO_FILL = (0, 255, 0)          # RGB: ego lane
OTHER_FILL = (255, 0, 0)        # RGB: every other lane


def cv2_color(name):
    """Convert a Matplotlib color name into the RGB tuple expected by OpenCV for drawing."""
    return tuple(int(255 * v) for v in mcolors.to_rgb(name))


LINE_COLORS = [cv2_color("magenta"), cv2_color("yellow"), cv2_color("blue")]


# functions to detect vehicles on the road
def vehicles_to_road(dets, M):
    """Project bbox ground-contact points (bottom-left/centre/right) to road coords via the BEV homography."""
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        xc, hw = 0.5 * (x1 + x2), 0.5 * FOOT_SHRINK * (x2 - x1)
        px = np.float32([[xc - hw, y2], [xc, y2], [xc + hw, y2]]).reshape(-1, 1, 2)
        bev = cv2.perspectiveTransform(px, M).reshape(-1, 2)
        x = X_RANGE[0] + bev[:, 0] / PX_PER_M
        z = Z_RANGE[1] - bev[:, 1] / PX_PER_M
        d.update(x_l=float(x[0]), x_r=float(x[2]), z=float(z[1]))
    return dets


def lead_vehicle(vehicles, left, right):
    """Closest vehicle whose footprint overlaps the ego lane at its own depth, or None."""
    hits = [v for v in vehicles
            if v["z"] > 0 and v["x_r"] > x_at(left, v["z"]) and v["x_l"] < x_at(right, v["z"])]
    return min(hits, key=lambda v: v["z"], default=None)


def draw_lead(img, M_inv, lead, left, right):
    """Draw the lead bbox and the distance lines across the ego lane (in place)."""
    if lead is None:
        return
    x1, y1, x2, y2 = lead["box"].astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 128, 0), 2)
    for z, col in ((lead["z"], (255, 0, 0)), (lead["z"] - D_MIN_M, (255, 255, 0))):
        zz = np.full(2, z)
        xx = np.array([x_at(left, z), x_at(right, z)])
        cv2.polylines(img, [bev_points_on_image(xx, zz, M_inv).astype(np.int32)], False, col, 3, cv2.LINE_AA)

def line_gap(l1, l2, n=8):
    """Mean lateral distance between two lines over the BEV depth."""
    z = np.linspace(Z_RANGE[0], Z_RANGE[1], n)
    return float(np.mean(np.abs(x_at(l1, z) - x_at(l2, z))))


def has_paint_evidence(c, min_frac=SNAP_MIN_RIDGE_FRAC):
    span = Z_RANGE[1] - Z_RANGE[0]
    return c["paint"] >= PAINT_MIN_FRAC and max(c["ridge_m"], c.get("color_m", 0.0)) >= min_frac * span


def cluster_by_gap(lines, max_gap):
    """Anchored clusters (no chaining): a line joins a cluster only if it is within max_gap of the cluster's FIRST (leftmost) line."""
    clusters = []
    for ln in sorted(lines, key=x_at):
        if clusters and line_gap(ln, clusters[-1][0]) < max_gap:
            clusters[-1].append(ln)
        else:
            clusters.append([ln])
    return clusters


def ego_edge_contrast(c, ego_prob, half_w_m=EGO_EDGE_HALF_W_M):
    """|mean ego_prob just right of the line - just left of it|, along the BEV depth. High = the line is an ego-lane border."""
    h, w = ego_prob.shape
    z = np.arange(Z_RANGE[0] + COV_BIN_M / 2, Z_RANGE[1], COV_BIN_M)
    rows = np.clip(((Z_RANGE[1] - z) * PX_PER_M).astype(int), 0, h - 1)
    cols = (x_at(c, z) - X_RANGE[0]) * PX_PER_M
    d = int(round(half_w_m * PX_PER_M))
    left, right = [], []
    for r, col in zip(rows, cols):
        ci = int(col)
        if ci - d < 0 or ci + d + 1 > w:
            continue
        left.append(ego_prob[r, ci - d:ci].mean())
        right.append(ego_prob[r, ci + 1:ci + d + 1].mean())
    if len(left) < 10:
        return 0.0
    return float(abs(np.mean(right) - np.mean(left)))


def pick_cluster_line(group, ego_prob=None):
    """
    Representative line of an interior cluster: the one with the sharpest ego_prob step among the lines
    with enough depth coverage; if the network shows no clear step, the highest line_score (old behaviour).
    """
    best_score = max(group, key=line_score)
    if ego_prob is None:
        return best_score
    span = Z_RANGE[1] - Z_RANGE[0]
    eligible = [c for c in group if c["cov_m"] >= NARROW_MIN_COV_FRAC * span]
    if not eligible:
        return best_score
    best = max(eligible, key=lambda c: ego_edge_contrast(c, ego_prob))
    return best if ego_edge_contrast(best, ego_prob) >= EGO_EDGE_MIN_CONTRAST else best_score


def select_boundaries(cands, left_edge, right_edge, ego_prob=None):
    """
    One boundary per cluster of nearby lines:
    - cluster around a road edge: best painted line with ridge evidence, else the road edge itself;
    - interior cluster of painted lines: the one with the highest line_score.
    """
    edges = [e for e in (left_edge, right_edge) if e]
    span = Z_RANGE[1] - Z_RANGE[0]
    pool = [c for c in filter_parallel(cands, edges) if c["paint"] >= PAINT_MIN_FRAC]

    # 1) edge clusters (anchored on the edge, so painted lines cannot chain away from it)
    keep, used = [], set()
    for e in edges:
        group = [c for c in pool if id(c) not in used and line_gap(c, e) < MERGE_DIST_M]
        used.update(id(c) for c in group)
        painted = [c for c in group if line_gap(c, e) < SNAP_MAX_DIST_M and has_paint_evidence(c, EDGE_SNAP_RIDGE_FRAC)]
        if painted:
            best = max(painted, key=line_score)
            keep.append(dict(best, edge=True, snapped=True, n_inl=int(best["inliers"].sum())))
        else:
            keep.append(e)

    # 2) interior clusters
    def narrow_ok(c):
        near_edge = any(abs(x_at(c) - x_at(e)) < MIN_LANE_WIDTH_M for e in edges)
        return not near_edge or c["cov_m"] >= NARROW_MIN_COV_FRAC * span

    rest = [c for c in pool if id(c) not in used and narrow_ok(c)]
    if left_edge:
        rest = [c for c in rest if x_at(c) > x_at(left_edge)]
    if right_edge:
        rest = [c for c in rest if x_at(c) < x_at(right_edge)]
        winners = [pick_cluster_line(g, ego_prob) for g in cluster_by_gap(rest, MIN_LANE_WIDTH_M)]
    if winners:
        top = max(line_score(w) for w in winners)
        keep += [w for w in winners if line_score(w) >= INTERIOR_MIN_SCORE_FRAC * top]
    keep.sort(key=x_at)

    # an estimated edge is only a guess: if a painted line already bounds the road on the non-ego side, drop it
    if len(keep) >= 3:
        if keep[0].get("estimated") and not keep[1].get("edge") and x_at(keep[1], EGO_CHECK_Z_M) < 0:
            keep = keep[1:]
        if len(keep) >= 3 and keep[-1].get("estimated") and not keep[-2].get("edge") and x_at(keep[-2], EGO_CHECK_Z_M) > 0:
            keep = keep[:-1]
    return keep


def restrict_to_edges(pts_dict, edge_l, edge_r, band=EDGE_BAND_M):
    """Keep only points lying inside the valid road band between the left and right edges."""
    keep = np.ones(len(pts_dict["x"]), bool)
    z = pts_dict["z"]
    if edge_l:
        keep &= pts_dict["x"] >= edge_l["a"] * z + edge_l["b"] - band
    if edge_r:
        keep &= pts_dict["x"] <= edge_r["a"] * z + edge_r["b"] + band
    return {k: v[keep] for k, v in pts_dict.items()}


# ---------------- binary image -> road coordinates ----------------
def points_in_meters(img_thresh, px_per_m=PX_PER_M, x_range=X_RANGE, z_range=Z_RANGE, margin_px=MARGIN_PX, bottom_margin_px=BOTTOM_MARGIN_PX):
    """Convert a thresholded bird's-eye-view mask into meter-based x/z coordinates for lane fitting."""
    height, width = img_thresh.shape
    row, column = np.nonzero(img_thresh)
    ok = ((row >= margin_px) & (row < height - bottom_margin_px) & (column >= margin_px) & (column < width - margin_px))
    row, column = row[ok], column[ok]
    x = x_range[0] + column / px_per_m
    z = z_range[1] - row / px_per_m
    return dict(x=x, z=z, row=row, column=column)


def road_edge_points(road, safe, direction, mode="strict"):
    """
    Collect boundary pixels on one side of the vehicle to estimate the road edge under strict,
    tolerant, or extreme fallback behavior depending on visibility and road mask gaps.
    """
    h, w = road.shape
    c = w // 2
    max_gap = int(ROAD_GAP_M * PX_PER_M)
    xs, zs = [], []
    for r in range(h):
        if mode == "extreme":
            cols = np.flatnonzero(road[r])
            if len(cols) == 0:
                continue
            col = cols[-1] + 1 if direction > 0 else cols[0] - 1
            if not (0 <= col < w) or direction * (col - c) <= 0 or not safe[r, col]:
                continue
        else:
            seg = road[r, c:] if direction > 0 else road[r, :c + 1][::-1]
            if seg[0]:
                start = 0
            elif mode == "tolerant":
                hits = np.flatnonzero(seg)
                if len(hits) == 0 or hits[0] > max_gap:
                    continue
                start = hits[0]
            else:
                continue
            end = np.flatnonzero(~seg[start:])
            if len(end) == 0:
                continue
            k = start + end[0]
            col = c + k if direction > 0 else c - k
            if not safe[r, col]:
                continue
        xs.append(X_RANGE[0] + col / PX_PER_M)
        zs.append(Z_RANGE[1] - r / PX_PER_M)
    return np.column_stack((xs, zs)) if xs else np.empty((0, 2))


def road_edge_candidate(road, safe, direction):
    """Fit a road-edge line from boundary points using a progressive fallback strategy: strict, tolerant, then extreme."""
    for mode in ("strict", "tolerant", "extreme"):
        p = road_edge_points(road, safe, direction, mode=mode)
        if len(p) < ROAD_EDGE_MIN_INLIERS:
            continue
        r = ransac(p, threshold=ROAD_EDGE_THRESHOLD_M, min_inliers=ROAD_EDGE_MIN_INLIERS)
        if r is None:
            continue
        r["length"] = coverage(p, r["inliers"]) * COV_BIN_M    # straight length (m) covered by the inliers
        r["n_inl"] = int(r["inliers"].sum())
        r["pts"] = p[r["inliers"]]             # support points, needed for the quadratic fit
        r["inliers"] = None
        r["edge"] = True
        r["mode"] = mode                   # for debugging
        return r
    return None


def road_edges(road, safe):
    """
    Estimate the left and right road edges, and if one edge is ambiguous, rebuild it with the stronger edge's slope.
    """
    edge_l = road_edge_candidate(road, safe, -1)
    edge_r = road_edge_candidate(road, safe, +1)
    if edge_l is None or edge_r is None or abs(edge_l["a"] - edge_r["a"]) <= EDGE_PARALLEL_TOL:
        return edge_l, edge_r

    left_is_strong = (edge_l["length"], edge_l["n_inl"]) >= (edge_r["length"], edge_r["n_inl"])
    strong, weak, side = (edge_l, edge_r, +1) if left_is_strong else (edge_r, edge_l, -1)

    p = road_edge_points(road, safe, side, mode=weak["mode"])    # boundary points of the weak side
    if len(p) == 0:
        return edge_l, edge_r
    x_out, z_out = p[np.argmax(side * p[:, 0])]                  # outermost point (max x right, min x left)
    a = strong["a"]
    fixed = dict(weak, a=a, b=float(x_out - a * z_out), estimated=True)

    z = np.array(Z_RANGE)                                        # the rebuilt edge must stay on its own side
    if np.any(side * ((fixed["a"] * z + fixed["b"]) - (strong["a"] * z + strong["b"])) <= 0):
        return edge_l, edge_r
    return (edge_l, fixed) if left_is_strong else (fixed, edge_r)


# ---------------- RANSAC (one line) ----------------
def n_iteration(p, w, n_points=2):
    """Estimate the number of RANSAC iterations needed to reach the target confidence for a given inlier ratio."""
    w_n = w ** n_points
    if w_n >= 1.0:
        return 1
    if w_n <= 0.0:
        return MAX_ITER
    return int(np.ceil(np.log(1.0 - p) / np.log(1.0 - w_n)))


def line_distance(points, a, b, c=0.0):
    """Orthogonal distance of each point to the curve x = c*z^2 + a*z + b (c=0 -> straight line)."""
    z = points[:, 1]
    slope = 2 * c * z + a
    return np.abs(points[:, 0] - (c * z ** 2 + a * z + b)) / np.sqrt(slope ** 2 + 1)


def line_fit(p1, p2, separation=MIN_SEPARATION_M, max_slope=MAX_SLOPE, a_ref=0.0):
    """Fit a candidate line from two points while enforcing a minimum z span and slope bounds."""
    (x1, z1), (x2, z2) = p1, p2
    dz = z2 - z1
    if abs(dz) < separation:
        return None
    a = (x2 - x1) / dz
    if abs(a - a_ref) > max_slope:
        return None
    return a, x1 - a * z1


def coverage(points, mask):
    """Count how many z-bins are covered by inliers, counting each depth bin only once."""
    zb = np.clip(((points[mask, 1] - Z_RANGE[0]) / COV_BIN_M).astype(int), 0, None)
    return int(np.count_nonzero(np.bincount(zb)))


def ridge_cov(c, grad, ref):
    """Measure the length of a line that lies on a strong ridge/paint response relative to the reference image."""
    h, w = grad.shape
    z = np.arange(Z_RANGE[0] + COV_BIN_M / 2, Z_RANGE[1], COV_BIN_M)
    rows = np.clip(((Z_RANGE[1] - z) * PX_PER_M).astype(int), 0, h - 1)
    cols = (x_at(c, z) - X_RANGE[0]) * PX_PER_M
    half = int(round(RIDGE_HALF_W_M * PX_PER_M))
    hit = 0
    for r, col in zip(rows, cols):
        c0, c1 = int(col) - half, int(col) + half + 1
        if c1 <= 0 or c0 >= w:
            continue
        if grad[r, max(c0, 0):c1].max() > RIDGE_STRONG_FRAC * ref:
            hit += 1
    return hit * COV_BIN_M


def ego_edge_cov(c, ego_prob):
    """Metres of the line where ego_prob steps sharply across it (ego-lane border). 0 if the network sees no ego lane."""
    if ego_prob is None or ego_prob.max() < EGO_PROB_MIN_PEAK:
        return 0.0
    h, w = ego_prob.shape
    z = np.arange(Z_RANGE[0] + COV_BIN_M / 2, Z_RANGE[1], COV_BIN_M)
    rows = np.clip(((Z_RANGE[1] - z) * PX_PER_M).astype(int), 0, h - 1)
    cols = (x_at(c, z) - X_RANGE[0]) * PX_PER_M
    d = int(round(EGO_EDGE_HALF_W_M * PX_PER_M))
    hit = 0
    for r, col in zip(rows, cols):
        ci = int(col)
        if ci - d < 0 or ci + d + 1 > w:
            continue
        left = ego_prob[r, ci - d:ci].mean()
        right = ego_prob[r, ci + 1:ci + d + 1].mean()
        if abs(right - left) >= EGO_EDGE_MIN_CONTRAST:
            hit += 1
    return hit * COV_BIN_M


def hypothesis_score(points, dist, threshold):
    """RANSAC score: z-coverage of the loose inliers + weighted z-coverage of the tight (stroke-like) inliers."""
    loose = dist <= threshold
    tight = dist <= TIGHT_THRESHOLD_M
    return coverage(points, loose) + TIGHT_WEIGHT * coverage(points, tight), loose


def ransac(points, threshold=INLINER_TRESHOLD_M, min_inliers=MIN_INLIERS, max_iter=MAX_ITER,
            p=SUCCESS_PROBABILITY, max_slope=MAX_SLOPE, sep_min=MIN_SEPARATION_M, seed=0, a_ref=0.0):
    """Use random sample consensus to find the best line model that explains the point cloud."""
    N = len(points)
    if N < max(2, min_inliers):
        return None
    rng = np.random.default_rng(seed)
    best, n_best, cov_best, best_mask = None, 0, 0, None
    k, n_iter, attempts = max_iter, 0, 0
    while n_iter < min(k, max_iter) and attempts < 50 * max_iter:
        attempts += 1
        i, j = rng.integers(0, N, size=2)
        line = line_fit(points[i], points[j], separation=sep_min, max_slope=max_slope, a_ref=a_ref)
        if line is not None:
            a, b = line
            n_iter += 1
            dist = line_distance(points, a, b)
            cov, inliers = hypothesis_score(points, dist, threshold)
            if cov > cov_best:
                best, n_best, cov_best, best_mask = (a, b), int(inliers.sum()), cov, inliers
                k = n_iteration(p, n_best / N)
    if best is None or n_best < min_inliers:
        return None
    a, b = np.polyfit(points[best_mask, 1], points[best_mask, 0], 1)
    if abs(a - a_ref) > max_slope:
        a, b = best                       # refit drifted out of the slope gate: keep the hypothesis that passed it
    best_mask = line_distance(points, a, b) <= threshold
    if int(best_mask.sum()) < min_inliers:
        return None
    return dict(a=float(a), b=float(b), inliers=best_mask, n_iter=n_iter)


def end_slopes_ok(c, a, a_ref, tol=CURVE_SLOPE_TOL):
    """A curve must stay roughly parallel to the road direction over the whole BEV depth, not only at mid depth."""
    return all(abs(2 * c * z + a - a_ref) <= tol for z in Z_RANGE)


def poly_fit_3(p1, p2, p3, sep_min=MIN_SEPARATION_M, max_curv=RANSAC_MAX_CURVATURE):
    """Exact parabola x = c*z^2 + a*z + b through 3 points; None if degenerate or too curved."""
    pts = sorted((p1, p2, p3), key=lambda p: p[1])
    z = np.array([p[1] for p in pts])
    x = np.array([p[0] for p in pts])
    if z[2] - z[0] < sep_min or min(z[1] - z[0], z[2] - z[1]) < 0.5 * sep_min:
        return None
    c, a, b = np.linalg.solve(np.vander(z, 3), x)      # columns: z^2, z, 1
    if abs(2 * c) > max_curv:
        return None
    return c, a, b


def ransac_poly(points, threshold=INLINER_TRESHOLD_M, min_inliers=MIN_INLIERS, max_iter=MAX_ITER,
                p=SUCCESS_PROBABILITY, max_slope=MAX_SLOPE, sep_min=MIN_SEPARATION_M,
                max_curv=RANSAC_MAX_CURVATURE, max_curve_slope=CURVE_SLOPE_TOL, seed=0, a_ref=0.0):
    """RANSAC for x = c*z^2 + a*z + b. Same scoring as ransac() (z-coverage), 3-point samples."""
    N = len(points)
    if N < max(3, min_inliers):
        return None
    rng = np.random.default_rng(seed)
    z_mid = 0.5 * (Z_RANGE[0] + Z_RANGE[1])
    best, n_best, cov_best, best_mask = None, 0, 0, None
    k, n_iter, attempts = max_iter, 0, 0
    while n_iter < min(k, max_iter) and attempts < 50 * max_iter:
        attempts += 1
        i, j, l = rng.integers(0, N, size=3)
        model = poly_fit_3(points[i], points[j], points[l], sep_min, max_curv)
        if model is None:
            continue
        c, a, b = model
        if abs(2 * c * z_mid + a - a_ref) > max_slope:      # slope gate at mid BEV
            continue
        if not end_slopes_ok(c, a, a_ref, max_curve_slope):  # slope gate at both BEV ends
            continue
        n_iter += 1
        inliers = line_distance(points, a, b, c) <= threshold
        cov = coverage(points, inliers)
        if cov > cov_best:
            best, n_best, cov_best, best_mask = (c, a, b), int(inliers.sum()), cov, inliers
            k = n_iteration(p, n_best / N, n_points=3)
    if best is None or n_best < min_inliers:
        return None
    c, a, b = np.polyfit(points[best_mask, 1], points[best_mask, 0], 2)
    if abs(2 * c) > max_curv or abs(2 * c * z_mid + a - a_ref) > max_slope or not end_slopes_ok(c, a, a_ref, max_curve_slope):
        c, a, b = best                                       # refit drifted out of the gates
    best_mask = line_distance(points, a, b, c) <= threshold
    if int(best_mask.sum()) < min_inliers:
        return None
    return dict(a=float(a), b=float(b), c=float(c), inliers=best_mask, n_iter=n_iter)


def support_score(points, r, threshold=INLINER_TRESHOLD_M):
    return hypothesis_score(points, line_distance(points, r["a"], r["b"], r.get("c", 0.0)), threshold)[0]


def ransac_best(points, max_curv=RANSAC_MAX_CURVATURE, **kw):
    lin = ransac(points, **kw)
    cur = ransac_poly(points, max_curv=max_curv, **kw)
    if cur is None or lin is None:
        return cur if lin is None else lin
    return cur if support_score(points, cur) >= CURVE_MIN_GAIN * support_score(points, lin) else lin


# ---------------- candidates ----------------
def x_at(line, z=None):
    """Lateral position of a line at depth z (scalar or array); supports the optional quadratic term c."""
    if z is None:
        z = 0.5 * (Z_RANGE[0] + Z_RANGE[1])
    return line.get("c", 0.0) * z ** 2 + line["a"] * z + line["b"]


def slope_at(line, z=None):
    """dx/dz of a line at depth z (default: mid BEV). For straight lines it is just a."""
    if z is None:
        z = 0.5 * (Z_RANGE[0] + Z_RANGE[1])
    return 2 * line.get("c", 0.0) * z + line["a"]


def find_candidates(points, n=N_CANDIDATES, threshold=INLINER_TRESHOLD_M, a_ref=0.0):
    """
    Iteratively extract the strongest lane candidates by running RANSAC, then removing nearby inliers
    so the same lane marking does not appear multiple times in the candidate list.
    """
    cands = []
    remaining = np.arange(len(points))
    for _ in range(n):
        if len(remaining) < MIN_INLIERS:
            break
        sub = points[remaining]
        r = ransac_best(sub, max_slope=SLOPE_TOL, a_ref=a_ref)
        if r is None:
            break
        full_mask = np.zeros(len(points), dtype=bool)
        full_mask[remaining[r["inliers"]]] = True
        r["inliers"] = full_mask
        cands.append(r)
        near = line_distance(sub, r["a"], r["b"], r.get("c", 0.0)) <= 2 * threshold
        remaining = remaining[~near]
    return cands


def line_score(c):
    return (c["cov_m"] + RIDGE_BONUS * c.get("ridge_m", 0.0) + COLOR_BONUS * c.get("color_m", 0.0) + TIGHT_BONUS * c.get("tight_m", 0.0) + EGO_EDGE_BONUS * c.get("ego_m", 0.0))

def filter_parallel(cands, edges):
    """Keep only candidate lines whose slope is close to the road edges or the dominant lane direction."""
    if not cands:
        return []
    if edges:
        a_ref = float(np.mean([e["a"] for e in edges]))          # road edges give the direction
    else:
        a = np.array([c["a"] for c in cands])
        w = np.array([c["inliers"].sum() for c in cands], float)
        o = np.argsort(a)
        cw = np.cumsum(w[o])
        a_ref = a[o][np.searchsorted(cw, cw[-1] / 2)]           # weighted median
    return [c for c in cands if abs(slope_at(c) - a_ref) < SLOPE_DELTA_MAX]


def lane_ego_score(left, right, ego_prob):
    """Soft IoU between the strip left..right (BEV) and the network ego-lane probability."""
    h, w = ego_prob.shape
    z = Z_RANGE[1] - (np.arange(h) + 0.5) / PX_PER_M
    c_l = ((x_at(left, z) - X_RANGE[0]) * PX_PER_M)[:, None]
    c_r = ((x_at(right, z) - X_RANGE[0]) * PX_PER_M)[:, None]
    cols = np.arange(w)[None, :]
    strip = (cols >= c_l) & (cols <= c_r)
    inter = float(ego_prob[strip].sum())
    return inter / max(strip.sum() + ego_prob.sum() - inter, 1.0)


def build_lanes(cands, left_edge, right_edge, ego_prob=None):
    """Assemble the lane boundaries into lanes and keep only the ego lane."""
    lines = select_boundaries(cands, left_edge, right_edge, ego_prob)
    def width_ok(l, r):
        return (l.get("edge") and r.get("edge")) or line_gap(l, r) <= MAX_LANE_WIDTH_M

    lanes = [[l, r, False] for l, r in zip(lines[:-1], lines[1:]) if width_ok(l, r)]
    zc = EGO_CHECK_Z_M
    idx = [i for i, (l, r, _) in enumerate(lanes) if x_at(l, zc) - EGO_AXIS_TOL_M < 0 < x_at(r, zc) + EGO_AXIS_TOL_M]
    if not idx:
        return []
    if len(idx) > 1:                      # the car straddles a line: let the network decide
        if ego_prob is not None:
            best = max(idx, key=lambda i: lane_ego_score(lanes[i][0], lanes[i][1], ego_prob))
        else:
            best = idx[-1]                # fallback: right-hand traffic -> rightmost lane
    else:
        best = idx[0]
    lanes[best][2] = True
    return [lanes[best]]


# ---------------- drawing ----------------
def bev_points_on_image(x, z, M_inv, x_range=X_RANGE, z_range=Z_RANGE, px_per_m=PX_PER_M):
    """Project BEV road coordinates (x, z in metres) onto the original image using the inverse homography."""
    col = (np.asarray(x) - x_range[0]) * px_per_m
    row = (z_range[1] - np.asarray(z)) * px_per_m
    points_px = np.column_stack((col, row)).astype(np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points_px, M_inv).reshape(-1, 2)


def line_on_image(line, M_inv, x_range=X_RANGE, z_range=Z_RANGE, px_per_m=PX_PER_M):
    """Project a fitted BEV line (x = c*z^2 + a*z + b, c optional) back onto the original image."""
    z = np.linspace(z_range[0], z_range[1], 100)
    x = line.get("c", 0.0) * z ** 2 + line["a"] * z + line["b"]
    return bev_points_on_image(x, z, M_inv, x_range, z_range, px_per_m)


def draw_lanes(img, M_inv, lanes, alpha=0.35, thickness=4):
    """Fill the detected lane regions and draw the boundary lines on the input image."""
    overlay = img.copy()
    for left, right, is_ego in lanes:
        area = np.vstack((line_on_image(left, M_inv), line_on_image(right, M_inv)[::-1])).astype(np.int32)
        cv2.fillPoly(overlay, [area], EGO_FILL if is_ego else OTHER_FILL)
    out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # unique boundaries, left to right: first left line + the right line of every lane
    boundaries = [lanes[0][0]] + [ln[1] for ln in lanes]
    for k, line in enumerate(boundaries):
        pts = line_on_image(line, M_inv).astype(np.int32)
        cv2.polylines(out, [pts], False, LINE_COLORS[k % len(LINE_COLORS)], thickness, cv2.LINE_AA)
    return out


# ---------------- curvature and prediction ----------------
def support_points(line, pts):
    """Return the (x, z) points supporting a boundary, or None if they are not reliable (estimated edge)."""
    if line.get("estimated"):
        return None
    if line.get("inliers") is not None:
        return pts[line["inliers"]]
    return line.get("pts")


def fit_boundary_poly(line, pts):
    """Quadratic x = c*z^2 + a*z + b on a boundary's support points, accepted only if it clearly beats the straight fit."""
    straight = np.array([line.get("c", 0.0), line["a"], line["b"]])
    if line.get("inliers") is not None:
        return straight          # RANSAC candidate: its own model (line or parabola) was already chosen by ransac_best --------------- HERE
    p = support_points(line, pts)
    if p is None or len(p) < MIN_POLY_POINTS or np.ptp(p[:, 1]) < MIN_POLY_SPAN_FRAC * (Z_RANGE[1] - Z_RANGE[0]):
        return straight

    # one sample per z-bin: dense clusters must not dominate the fit
    zb = np.floor((p[:, 1] - Z_RANGE[0]) / COV_BIN_M).astype(int)
    bins = np.unique(zb)
    z = np.array([p[zb == k, 1].mean() for k in bins])
    x = np.array([np.median(p[zb == k, 0]) for k in bins])
    if len(bins) < 10:
        return straight

    c, a, b = np.polyfit(z, x, 2)
    a1, b1 = np.polyfit(z, x, 1)
    rms_q = float(np.sqrt(np.mean((x - np.polyval([c, a, b], z)) ** 2)))
    rms_l = float(np.sqrt(np.mean((x - np.polyval([a1, b1], z)) ** 2)))

    if abs(2 * c) > POLY_MAX_CURVATURE or rms_l < POLY_STRAIGHT_RMS_M or rms_q > POLY_MIN_GAIN * rms_l:
        return straight
    return np.array([c, a, b])


def ego_centerline(ego, pts):
    """Centerline of the ego lane as x = c*z^2 + a*z + b (mean of the two boundary polynomials)."""
    coeffs = 0.5 * (fit_boundary_poly(ego[0], pts) + fit_boundary_poly(ego[1], pts))
    return dict(c=float(coeffs[0]), a=float(coeffs[1]), b=float(coeffs[2]))


def curvature_at(center, z):
    """Signed curvature [1/m] of x(z) at depth z (positive = curving right) and radius [m]."""
    slope = 2 * center["c"] * z + center["a"]
    kappa = 2 * center["c"] / (1 + slope ** 2) ** 1.5
    radius = 1.0 / abs(kappa) if abs(kappa) > 1e-6 else 1e6
    return float(kappa), float(min(radius, 1e6))


def lookahead_point(center, lookahead=LOOKAHEAD_M):
    """First point of the centerline at distance >= lookahead from the vehicle origin (x=0, z=EGO_Z_M)."""
    z = np.linspace(Z_RANGE[0], Z_RANGE[1], 200)
    x = center["c"] * z ** 2 + center["a"] * z + center["b"]
    ok = np.flatnonzero(np.hypot(x, z - EGO_Z_M) >= lookahead)
    i = int(ok[0]) if len(ok) else len(z) - 1
    return float(x[i]), float(z[i])


def draw_prediction(img, M_inv, center, target):
    """Draw the blue centerline and the red lookahead dot on the image (in place)."""
    cv2.polylines(img, [line_on_image(center, M_inv).astype(np.int32)], False, (0, 0, 255), 3, cv2.LINE_AA)
    u, v = bev_points_on_image([target[0]], [target[1]], M_inv)[0]
    cv2.circle(img, (int(u), int(v)), 8, (255, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, (int(u), int(v)), 10, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------- main function ----------------
def lane_fitting(pipe, img, vehicles=()):
    """Run the full lane-detection pipeline to estimate road boundaries, candidate lane markings, and final lanes."""
    safe = cv2.erode(pipe["valid"].astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
    edge_l, edge_r = road_edges(pipe["road"], safe)
    raw_edges = [e for e in (edge_l, edge_r) if e]
    a_ref = float(np.mean([e["a"] for e in raw_edges])) if raw_edges else 0.0

    pts_dict = restrict_to_edges(points_in_meters(pipe["thresh"]), edge_l, edge_r)
    pts = np.column_stack((pts_dict["x"], pts_dict["z"]))
    cands = find_candidates(pts, a_ref=a_ref)
    paint = (pipe["grad_thr"][pts_dict["row"], pts_dict["column"]] > 0) | (pipe["color"][pts_dict["row"], pts_dict["column"]] > 0)
    grad_ref = float(np.percentile(pipe["grad"][pipe["valid"]], RIDGE_REF_PCT))
    for c in cands:
        c["paint"] = float(paint[c["inliers"]].mean())
        c["cov_m"] = coverage(pts, c["inliers"]) * COV_BIN_M
        c["ridge_m"] = ridge_cov(c, pipe["grad"], grad_ref)
        c["color_m"] = ridge_cov(c, pipe["color"], 255.0) 
        tight = line_distance(pts, c["a"], c["b"], c.get("c", 0.0)) <= TIGHT_THRESHOLD_M
        c["tight_m"] = coverage(pts, tight) * COV_BIN_M
        c["ego_m"] = ego_edge_cov(c, pipe.get("ego_prob"))

    lanes = build_lanes(cands, edge_l, edge_r, pipe.get("ego_prob"))
    ego = next((ln for ln in lanes if ln[2]), None)

    out = dict(points=pts_dict, pts=pts, candidates=cands, edges=[e for e in (edge_l, edge_r) if e], lanes=lanes,
                ransac_sx=ego[0] if ego else None, ransac_dx=ego[1] if ego else None)
    if ego:
        center = ego_centerline(ego, pts)
        kappa, radius = curvature_at(center, Z_RANGE[0])
        target = lookahead_point(center)
        out.update(center=center, curvature=kappa, radius=radius, target=target)
        curved = [[dict(zip("cab", fit_boundary_poly(l, pts))) for l in ego[:2]] + [True]]
        out["overlay"] = draw_lanes(img, pipe["M_inv"], curved)
        draw_prediction(out["overlay"], pipe["M_inv"], center, target)
        lead = lead_vehicle(vehicles, ego[0], ego[1])
        out["lead"] = lead
        out["v_max"] = None if lead is None else max(0.0, (lead["z"] - D_MIN_M) / T_GAP_S)
        draw_lead(out["overlay"], pipe["M_inv"], lead, ego[0], ego[1])
    return out


def plot_bev_debug(img_thresh, fitting_out, out_path):
    """Save a BEV debug plot showing the road, candidates, and final fitted lane lines."""
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.imshow(img_thresh, cmap="gray", origin="upper", extent=(X_RANGE[0], X_RANGE[1], Z_RANGE[0], Z_RANGE[1]), aspect="auto")
    ax.axvline(0, color="yellow", linestyle="--", linewidth=1, label="vehicle centre")

    pts = fitting_out["pts"]
    for label, line_key, color in (("Left", "ransac_sx", "red"), ("Right", "ransac_dx", "cyan")):
        line = fitting_out[line_key]
        if line is None:
            ax.plot([], [], color=color, label=f"{label}: line not found")
            continue

        if line.get("inliers") is not None:
            inl = pts[line["inliers"]]
            ax.scatter(inl[:, 0], inl[:, 1], s=5, color=color, label=f"Inliers {label.lower()} ({len(inl)})")
            z_values = np.linspace(inl[:, 1].min(), inl[:, 1].max(), 100)
            style, name = "-", f"Line {label.lower()}"
        else:
            z_values = np.linspace(Z_RANGE[0], Z_RANGE[1], 100)
            style, name = "-.", f"Line {label.lower()} (road edge)"
        ax.plot(x_at(line, z_values), z_values, color=color, linewidth=2, linestyle=style, label=name)

    ax.set(xlabel="lateral x (m)", ylabel="forward z (m)", title="Bird-eye view: candidates and RANSAC inliers", xlim=X_RANGE, ylim=Z_RANGE)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_candidates(img, img_thresh, pts, cands, M_inv, out_bev, out_img):
    """Export a visualization of all candidate lane lines in BEV and in the original camera image."""
    z = np.linspace(Z_RANGE[0], Z_RANGE[1], 100)

    # 1) bird-eye view with all candidates
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.imshow(img_thresh, cmap="gray", origin="upper", extent=(X_RANGE[0], X_RANGE[1], Z_RANGE[0], Z_RANGE[1]), aspect="auto")
    ax.axvline(0, color="yellow", linestyle="--", linewidth=1)

    vis = img.copy()
    for k, c in enumerate(cands):
        name = CAND_COLORS[k % len(CAND_COLORS)]
        n_inl = 0
        if c.get("inliers") is not None:
            inl = pts[c["inliers"]]
            ax.scatter(inl[:, 0], inl[:, 1], s=4, color=name)
            n_inl = len(inl)
        tag = " (road edge)" if c.get("edge") else ""
        ax.plot(x_at(c, z), z, color=name, linewidth=2, label=f"cand {k}{tag}: a={c['a']:+.2f} b={c['b']:+.2f} inl={n_inl}")

        pts_px = line_on_image(c, M_inv).astype(np.int32)
        cv2.polylines(vis, [pts_px], False, cv2_color(name), 3)

    ax.set(xlabel="lateral x (m)", ylabel="forward z (m)", title="RANSAC candidates", xlim=X_RANGE, ylim=Z_RANGE)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_bev, dpi=150)
    plt.close(fig)

    # 2) original image with all candidates reprojected
    cv2.imwrite(out_img, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def print_candidate_stats(out):
    """Print the numbers that decide snapping and lane selection."""
    for k, c in enumerate(out["candidates"]):
        print(f"cand {k}: x@mid={x_at(c):+.2f} a={c['a']:+.2f} cov={c['cov_m']:.1f}m "
                f"paint={c['paint']:.2f} ridge={c['ridge_m']:.1f}m inl={int(c['inliers'].sum())} ego={c['ego_m']:.1f}m")
    for l, r, _ in out["lanes"]:
        for name, ln in (("left", l), ("right", r)):
            print(f"ego {name}: x@mid={x_at(ln):+.2f} edge={ln.get('edge', False)} snapped={ln.get('snapped', False)}")