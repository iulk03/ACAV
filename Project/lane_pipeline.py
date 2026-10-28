import cv2
import matplotlib.pyplot as plt
import numpy as np
import lane_fitting as lf

#import road_classifier as nc
import net_classifier as nc

X_RANGE = (-7.0, 7.0)       # road width in metres
Z_RANGE = (6.5, 20.0)       # road depth in metres
PX_PER_M = 30               # bird-eye view resolution (1 px = 1/30 m)
GAUSS_KSIZE = 5             # gaussian kernel size (noise reduction)
PERCENTILE = 97             # keep the strongest 3% gradient pixels (per block)
MIN_GRAD = 0.0              # absolute minimum gradient threshold
BLOCK_PX = 60               # column block used for the local percentile
MIN_BLOCK_PIXELS = 100      # below this many region pixels in a block, use the global threshold
ROAD_ERODE_PX = 15          # ~0.5 m: keeps painted-line candidates away from the road border
MIN_ROAD_FRAC = 0.10        # if the road mask covers less than this, consider it failed
LINE_WIDTH_PX = 13      # ~0.43 m: anything wider than this is not paint
LINE_MIN_LEN_PX = 21    # ~0.7 m: anything shorter vertically is not a marking
MIN_WEDGE_M = 2.5       # sanity bounds on the road width
MAX_WEDGE_M = 12.0

USE_OLD_SOBEL = True        # prova: unisce al segnale ridge anche il vecchio Sobel


# ---------------- calibration and geometry ----------------
def read_calib(path):
    calib = {}
    with open(path) as f:
        for row in f:
            if ":" in row:
                name, values = row.split(":", 1)
                calib[name.strip()] = np.array(values.split(), dtype=float)
    return calib


def project_to_image(calib, x, z):
    """Project the road point (x, 0, z) onto the image: returns (u, v) in pixels."""
    P2 = calib["P2"].reshape(3, 4)
    R0_rect = calib["R0_rect"].reshape(3, 3)
    T = calib["Tr_cam_to_road"].reshape(3, 4)
    R, t = T[:, :3], T[:, 3]
    p_cam = R.T @ (np.array([x, 0.0, z]) - t)
    u, v, w = P2 @ np.append(R0_rect @ p_cam, 1.0)
    return [u / w, v / w]


def z_at_image_bottom(calib, img_h, x=0.0, z_lo=1.0, z_hi=20.0):
    """Road depth z [m] whose ground point projects onto the last image row (bisection, v decreases with z)."""
    for _ in range(40):
        mid = 0.5 * (z_lo + z_hi)
        if project_to_image(calib, x, mid)[1] > img_h - 1:   # still below the image -> go farther
            z_lo = mid
        else:
            z_hi = mid
    return 0.5 * (z_lo + z_hi)


def compute_trapezoid(calib, x_range=X_RANGE, z_range=Z_RANGE):
    """Project the road rectangle (x, z) onto the image: returns the 4 trapezoid corners in pixels."""
    return np.float32([
        project_to_image(calib, x_range[0], z_range[0]),
        project_to_image(calib, x_range[1], z_range[0]),
        project_to_image(calib, x_range[1], z_range[1]),
        project_to_image(calib, x_range[0], z_range[1]),
    ])


def apply_roi(img, trapezoid):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [trapezoid.astype(np.int32)], 255)
    return cv2.bitwise_and(img, img, mask=mask)


# NOTE: the name "brid_eye_view" is kept on purpose: road_classifier.py calls lp.brid_eye_view
def brid_eye_view(img, trapezoid, x_range=X_RANGE, z_range=Z_RANGE, px_per_m=PX_PER_M):
    width = int((x_range[1] - x_range[0]) * px_per_m)
    height = int((z_range[1] - z_range[0]) * px_per_m)
    trapezoid_dst = np.float32([[0, height], [width, height], [width, 0], [0, 0]])
    M = cv2.getPerspectiveTransform(trapezoid, trapezoid_dst)
    M_inv = cv2.getPerspectiveTransform(trapezoid_dst, trapezoid)
    bev = cv2.warpPerspective(img, M, (width, height))
    return bev, M, M_inv


# ---------------- gradient segmentation (painted lines) ----------------
def gaussian_filter(img, ksize=GAUSS_KSIZE):
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.GaussianBlur(grey, (ksize, ksize), 0)


def gradient_filter(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def threshold(grad, region, percentile=PERCENTILE, min_grad=MIN_GRAD, block=BLOCK_PX):
    """
    Per-block percentile threshold computed ONLY on the pixels inside `region`
    (so grass/hedge texture does not raise the threshold and hide the painted lines).
    """
    out = np.zeros(grad.shape, np.uint8)
    if not region.any():
        return out
    global_thr = max(np.percentile(grad[region], percentile), min_grad)
    for c in range(0, grad.shape[1], block):
        ok = region[:, c:c + block]
        blk = grad[:, c:c + block]
        if ok.sum() >= MIN_BLOCK_PIXELS:
            thr = max(np.percentile(blk[ok], percentile), min_grad)
        else:
            thr = global_thr
        out[:, c:c + block] = ((blk > thr) & ok).astype(np.uint8) * 255
    return out


def slanted_kernel(a, length=LINE_MIN_LEN_PX):
    """Thin line kernel following x = a*z in the BEV (row 0 = far end)."""
    w = int(np.ceil(abs(a) * length)) + 1
    k = np.zeros((length, w), np.uint8)
    top, bottom = ((w - 1, 0), (0, length - 1)) if a > 0 else ((0, 0), (w - 1, length - 1))
    cv2.line(k, top, bottom, 1, 1)
    return k


def line_ridge(gray, valid, a=0.0):
    k_w = cv2.getStructuringElement(cv2.MORPH_RECT, (LINE_WIDTH_PX, 1))
    ridge = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k_w)
    ridge = cv2.morphologyEx(ridge, cv2.MORPH_OPEN, slanted_kernel(a))
    return ridge * valid


def road_wedge(edge_l, edge_r, shape):
    """Area between two fitted road edges, extrapolated over the whole BEV height."""
    h, w = shape
    wedge = np.zeros((h, w), bool)
    if edge_l is None or edge_r is None:
        return wedge                     # one edge alone would cover half the image: do nothing
    z = Z_RANGE[1] - (np.arange(h) + 0.5) / PX_PER_M
    x_l = edge_l["a"] * z + edge_l["b"]
    x_r = edge_r["a"] * z + edge_r["b"]
    width = x_r - x_l
    if width.min() < MIN_WEDGE_M or width.max() > MAX_WEDGE_M:
        return wedge                     # implausible geometry: the extrapolation is wrong
    cols = np.arange(w)[None, :]
    c_l = ((x_l - X_RANGE[0]) * PX_PER_M)[:, None]
    c_r = ((x_r - X_RANGE[0]) * PX_PER_M)[:, None]
    return (cols >= c_l) & (cols <= c_r)


# ---------------- main pipeline ----------------
def lane_pipeline(calib_path, img, out_path=None):
    calib = read_calib(calib_path)
    trapezoid = compute_trapezoid(calib)
    bev, M, M_inv = brid_eye_view(apply_roi(img, trapezoid), trapezoid)
    h, w = bev.shape[:2]

    # BEV pixels that really come from the image (the trapezoid can stick out of it)
    valid = cv2.warpPerspective(np.full(img.shape[:2], 255, np.uint8), M, (w, h),
                                flags=cv2.INTER_NEAREST) > 0

    gray = gaussian_filter(bev)

    # 1) classifier branch: road mask, road contour, colour edges
    road_u8 = nc.road_mask_bev(bev, valid)              # uint8 0/255
    ego_prob = nc.ego_prob_bev(bev, valid)
    road_ok = (road_u8 > 0).mean() > MIN_ROAD_FRAC
    edge_l = edge_r = None                              # defined in both branches
    if road_ok:
        road = road_u8 > 0
        safe = cv2.erode(valid.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
        edge_l, edge_r = lf.road_edges(road, safe)
        wedge = road_wedge(edge_l, edge_r, road.shape)
        region = cv2.erode((road | wedge).astype(np.uint8), np.ones((ROAD_ERODE_PX, ROAD_ERODE_PX), np.uint8)) > 0
        road_edge = nc.road_edge(road_u8, valid)
    else:
        road_u8 = np.zeros_like(road_u8)
        road = np.zeros(road_u8.shape, dtype=bool)
        region = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        road_edge = np.zeros_like(road_u8)

    # 2) gradient branch: painted-line candidates, ridge kernel oriented like the road
    edges = [e for e in (edge_l, edge_r) if e]
    a_ref = float(np.mean([e["a"] for e in edges])) if edges else 0.0
    grad = line_ridge(gray, valid, a=a_ref).astype(np.float64)

    #prova per sobel
    grad_thr = threshold(grad, region)                   # solo ridge: serve al punteggio "paint"
    grad_thr_all = grad_thr
    if USE_OLD_SOBEL:
        sobel = gradient_filter(gray)
        sobel_thr = threshold(sobel, region)
        grad_thr_all = cv2.bitwise_or(grad_thr, sobel_thr)

    chroma = nc.chroma_edges(bev, valid)
    thresh = cv2.bitwise_or(grad_thr_all, chroma)
    #fin e prova sobel

    if out_path is not None:
        vis = img.copy()
        cv2.polylines(vis, [trapezoid.astype("int32")], True, (255, 0, 0), 3)
        save_panels([
            ("ROI", vis, None),
            ("Bird-eye", bev, None),
            ("Gaussian", gray, "gray"),
            ("Gradient", grad, "gray"),
            ("Gradient threshold (road only)", grad_thr, "gray"),
            ("Road mask", road_u8, "gray"),
            ("Road edge", road_edge, "gray"),
            ("Colour edges", chroma, "gray"),
            ("Final threshold", thresh, "gray"),
        ], out_path)

    return dict(M=M, M_inv=M_inv, thresh=thresh, road_edge=road_edge, road=road, valid=valid, gray=gray, grad=grad, grad_thr=grad_thr, ego_prob=ego_prob)


def save_panels(panels, out_path, ncols=3):
    """Save a grid of (title, image, cmap) panels to out_path."""
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, data, cmap) in zip(axes, panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)