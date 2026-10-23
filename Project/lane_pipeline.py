import cv2
import matplotlib.pyplot as plt
import numpy as np

import road_classifier as rc

X_RANGE = (-7.0, 7.0)       # road width in metres
Z_RANGE = (7.0, 20.0)       # road depth in metres
PX_PER_M = 30               # bird-eye view resolution (1 px = 1/30 m)
GAUSS_KSIZE = 5             # gaussian kernel size (noise reduction)
PERCENTILE = 97             # keep the strongest 3% gradient pixels (per block)
MIN_GRAD = 0.0              # absolute minimum gradient threshold
BLOCK_PX = 60               # column block used for the local percentile
MIN_BLOCK_PIXELS = 100      # below this many region pixels in a block, use the global threshold
ROAD_ERODE_PX = 15          # ~0.5 m: keeps painted-line candidates away from the road border
MIN_ROAD_FRAC = 0.10        # if the road mask covers less than this, consider it failed


# ---------------- calibration and geometry ----------------
def read_calib(path):
    calib = {}
    with open(path) as f:
        for row in f:
            if ":" in row:
                name, values = row.split(":", 1)
                calib[name.strip()] = np.array(values.split(), dtype=float)
    return calib


def compute_trapezoid(calib, x_range=X_RANGE, z_range=Z_RANGE):
    """Project the road rectangle (x, z) onto the image: returns the 4 trapezoid corners in pixels."""
    P2 = calib["P2"].reshape(3, 4)
    R0_rect = calib["R0_rect"].reshape(3, 3)
    T = calib["Tr_cam_to_road"].reshape(3, 4)
    R, t = T[:, :3], T[:, 3]

    def project(x, z):
        p_road = np.array([x, 0.0, z])
        p_cam = R.T @ (p_road - t)
        p_cam_rect = R0_rect @ p_cam
        u, v, w = P2 @ np.append(p_cam_rect, 1.0)
        return [u / w, v / w]

    return np.float32([
        project(x_range[0], z_range[0]),
        project(x_range[1], z_range[0]),
        project(x_range[1], z_range[1]),
        project(x_range[0], z_range[1]),
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


# ---------------- main pipeline ----------------
def lane_pipeline(calib_path, img, out_path=None):
    calib = read_calib(calib_path)
    trapezoid = compute_trapezoid(calib)
    bev, M, M_inv = brid_eye_view(apply_roi(img, trapezoid), trapezoid)
    h, w = bev.shape[:2]

    # BEV pixels that really come from the image (the trapezoid can stick out of it)
    valid = cv2.warpPerspective(np.full(img.shape[:2], 255, np.uint8), M, (w, h),
                                flags=cv2.INTER_NEAREST) > 0

    # 1) gradient branch: painted-line candidates
    gray = gaussian_filter(bev)
    grad = gradient_filter(gray) * valid

    # 2) classifier branch (from swami): road mask, road contour, colour edges
    road_u8 = rc.road_mask_bev(bev, valid)              # uint8 0/255
    road_ok = (road_u8 > 0).mean() > MIN_ROAD_FRAC
    if road_ok:
        road = road_u8 > 0
        # painted lines are searched only well inside the road
        region = cv2.erode(road.astype(np.uint8), np.ones((ROAD_ERODE_PX, ROAD_ERODE_PX), np.uint8)) > 0
        road_edge = rc.road_edge(road_u8, valid)        # road contour, used as separate edge candidates
    else:
        road_u8 = np.zeros_like(road_u8)                # failed mask: no road-edge candidates either
        road = np.zeros(road_u8.shape, dtype=bool)
        region = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        road_edge = np.zeros_like(road_u8)

    grad_thr = threshold(grad, region)
    chroma = rc.chroma_edges(bev, valid)                # colour edges: asphalt/grass, painted lines
    thresh = cv2.bitwise_or(grad_thr, chroma)           # what RANSAC sees (road edge stays separate)

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

    return dict(M=M, M_inv=M_inv, thresh=thresh, road_edge=road_edge,
                road=road, valid=valid, gray=gray, grad=grad)


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