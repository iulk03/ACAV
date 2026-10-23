import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import cv2
from lane_pipeline import X_RANGE, Z_RANGE, PX_PER_M

# ---------------- RANSAC parameters ----------------
MARGIN_PX = 6               # ignore pixels too close to the image border
INLINER_TRESHOLD_M = 0.20   # max distance to be an inlier
MIN_SEPARATION_M = 3.0      # min z distance between the two points defining a line
SUCCESS_PROBABILITY = 0.99
MAX_ITER = 500
MIN_INLIERS = 100
MAX_SLOPE = 0.5             # max slope of a single line (x = a*z + b)
SNAP_MIN_SCORE = 0.3        # fraction of the line length that must look like a road boundary
SNAP_OFFSET_M = 0.3

# ---------------- candidate and lane parameters ----------------
N_CANDIDATES = 10           # how many candidate lines to search for
SLOPE_DELTA_MAX = 0.5       # max slope difference to be considered parallel
MERGE_DIST_M = 1.0          # lines closer than this (in x) are merged
ROAD_EDGE_THRESHOLD_M = 0.25
ROAD_EDGE_MIN_INLIERS = 40
MIN_SUPPORT_FRAC = 0.2   # painted lines with fewer inliers than 20% of the best one are dropped
MERGE_DIST_M = 1.0          # painted line vs road edge
MIN_LANE_WIDTH_M = 2.0      # painted line vs painted line

CAND_COLORS = ["red", "cyan", "lime", "magenta", "orange", "dodgerblue", "gold", "deeppink", "white", "mediumpurple"]


def boundary_score(c, road, side):
    """Fraction of the line where the inner side is road and the outer side is not."""
    h, w = road.shape
    z = np.linspace(Z_RANGE[0] + 0.5, Z_RANGE[1] - 0.5, 60)
    x = c["a"] * z + c["b"]
    rows = ((Z_RANGE[1] - z) * PX_PER_M).astype(int)
    c_in = ((x - side * SNAP_OFFSET_M - X_RANGE[0]) * PX_PER_M).astype(int)
    c_out = ((x + side * SNAP_OFFSET_M - X_RANGE[0]) * PX_PER_M).astype(int)
    ok = (rows >= 0) & (rows < h) & (c_in >= 0) & (c_in < w) & (c_out >= 0) & (c_out < w)
    if not ok.any():
        return 0.0
    hit = road[rows[ok], c_in[ok]] & ~road[rows[ok], c_out[ok]]
    return float(hit.sum()) / len(z)          # normalised by the full length, not only the visible part


def snap_edge(edge, cands, road, side):
    """Replace the RANSAC road edge with the innermost painted line that behaves like a boundary."""
    on_side = [c for c in cands if side * x_at(c) > 0]
    on_side.sort(key=lambda c: side * x_at(c))                 # innermost first
    for c in on_side:
        if boundary_score(c, road, side) >= SNAP_MIN_SCORE:
            if edge is None or side * x_at(c) < side * x_at(edge):
                return dict(c, edge=True)
            break
    return edge


def cv2_color(name):
    """Matplotlib colour name -> RGB tuple 0-255 for cv2 (the images are RGB)."""
    return tuple(int(255 * v) for v in mcolors.to_rgb(name))


# ---------------- binary image -> road coordinates ----------------
def points_in_meters(img_thresh, px_per_m=PX_PER_M, x_range=X_RANGE, z_range=Z_RANGE, margin_px=MARGIN_PX):
    height, width = img_thresh.shape
    row, column = np.nonzero(img_thresh)
    ok = (row >= margin_px) & (row < height - margin_px) & (column >= margin_px) & (column < width - margin_px)
    row, column = row[ok], column[ok]
    x = x_range[0] + column / px_per_m
    z = z_range[1] - row / px_per_m
    return dict(x=x, z=z)


def road_edge_points(road, safe, direction):
    """First road -> non-road transition going outward from the vehicle column, per row."""
    h, w = road.shape
    c = w // 2
    xs, zs = [], []
    for r in range(h):
        row = road[r]
        if not row[c]:
            continue
        if direction > 0:
            nz = np.flatnonzero(~row[c:])
            col = c + nz[0] if len(nz) else None
        else:
            nz = np.flatnonzero(~row[:c + 1][::-1])
            col = c - nz[0] if len(nz) else None
        if col is None or not safe[r, col]:
            continue                      # road touches the border / boundary is fake
        xs.append(X_RANGE[0] + col / PX_PER_M)
        zs.append(Z_RANGE[1] - r / PX_PER_M)
    return np.column_stack((xs, zs)) if xs else np.empty((0, 2))


# ---------------- RANSAC (one line) ----------------
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


# ---------------- candidates ----------------
def x_at(line, z=None):
    """Lateral position of a line at a z inside the range (not at z=0, which is extrapolated)."""
    if z is None:
        z = 0.5 * (Z_RANGE[0] + Z_RANGE[1])
    return line["a"] * z + line["b"]


def find_candidates(points, n=N_CANDIDATES, threshold=INLINER_TRESHOLD_M):
    """
    Sequential RANSAC over ALL points: find the best line, remove the points near it
    (band 2*threshold, so the other edge of the same line is not picked again), and repeat.
    The 'inliers' mask of each candidate refers to the original array.
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


def road_edge_candidate(road, safe, direction):
    p = road_edge_points(road, safe, direction)
    if len(p) < ROAD_EDGE_MIN_INLIERS:
        return None
    r = ransac(p, threshold=ROAD_EDGE_THRESHOLD_M, min_inliers=ROAD_EDGE_MIN_INLIERS)
    if r is None:
        return None
    r["inliers"] = None
    r["edge"] = True
    return r


def _rank(c):
    """Road edges always win a merge; among painted lines, the one with more inliers wins."""
    return (1, 0) if c.get("edge") else (0, int(c["inliers"].sum()))


def filter_parallel(cands, edges):
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
    return [c for c in cands if abs(c["a"] - a_ref) < SLOPE_DELTA_MAX]


def merge_close(lines):
    out = []
    for c in sorted(lines, key=x_at):
        if out:
            gap = x_at(c) - x_at(out[-1])
            both_painted = not c.get("edge") and not out[-1].get("edge")
            if gap < (MIN_LANE_WIDTH_M if both_painted else MERGE_DIST_M):
                if _rank(c) > _rank(out[-1]):
                    out[-1] = c
                continue
        out.append(c)
    return out


def build_lanes(cands, left_edge, right_edge):
    edges = [e for e in (left_edge, right_edge) if e]
    painted = filter_parallel(cands, edges)
    if painted:
        best = max(int(c["inliers"].sum()) for c in painted)
        painted = [c for c in painted if c["inliers"].sum() >= MIN_SUPPORT_FRAC * best]
    if left_edge:
        painted = [c for c in painted if x_at(c) > x_at(left_edge) + MERGE_DIST_M]
    if right_edge:
        painted = [c for c in painted if x_at(c) < x_at(right_edge) - MERGE_DIST_M]
    lines = merge_close(painted + edges)
    z_ego = Z_RANGE[0]
    return [[l, r, x_at(l, z_ego) < 0 < x_at(r, z_ego)] for l, r in zip(lines[:-1], lines[1:])]


# ---------------- drawing ----------------
def line_on_image(line, M_inv, x_range=X_RANGE, z_range=Z_RANGE, px_per_m=PX_PER_M):
    a, b = line["a"], line["b"]
    z = np.linspace(z_range[0], z_range[1], 100)
    x = a * z + b
    col = (x - x_range[0]) * px_per_m
    row = (z_range[1] - z) * px_per_m
    points_px = np.column_stack((col, row)).astype(np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points_px, M_inv).reshape(-1, 2)


def draw_lanes(img, M_inv, lanes, alpha=0.35):
    """Paint the whole carriageway green: from the leftmost to the rightmost line."""
    left, right = lanes[0][0], lanes[-1][1]
    area = np.vstack((line_on_image(left, M_inv), line_on_image(right, M_inv)[::-1])).astype(np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [area], (0, 255, 0))
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


# ---------------- main function ----------------
def lane_fitting(pipe, img):
    pts_dict = points_in_meters(pipe["thresh"])
    pts = np.column_stack((pts_dict["x"], pts_dict["z"]))

    cands = find_candidates(pts)
    safe = cv2.erode(pipe["valid"].astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
    edge_l = road_edge_candidate(pipe["road"], safe, -1)
    edge_r = road_edge_candidate(pipe["road"], safe, +1)
    edge_l = snap_edge(edge_l, cands, pipe["road"], -1)
    edge_r = snap_edge(edge_r, cands, pipe["road"], +1)
    lanes = build_lanes(cands, edge_l, edge_r)
    ego = next((ln for ln in lanes if ln[2]), None)

    out = dict(points=pts_dict, pts=pts, candidates=cands, edges=[e for e in (edge_l, edge_r) if e], lanes=lanes,
                ransac_sx=ego[0] if ego else None, ransac_dx=ego[1] if ego else None)
    if lanes:
        out["overlay"] = draw_lanes(img, pipe["M_inv"], lanes)
    return out


def plot_bev_debug(img_thresh, fitting_out, out_path):
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
        ax.plot(line["a"] * z_values + line["b"], z_values,
                color=color, linewidth=2, linestyle=style, label=name)

    ax.set(xlabel="lateral x (m)", ylabel="forward z (m)", title="Bird-eye view: candidates and RANSAC inliers", xlim=X_RANGE, ylim=Z_RANGE)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_candidates(img, img_thresh, pts, cands, M_inv, out_bev, out_img):
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
        ax.plot(c["a"] * z + c["b"], z, color=name, linewidth=2,
                label=f"cand {k}{tag}: a={c['a']:+.2f} b={c['b']:+.2f} inl={n_inl}")

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