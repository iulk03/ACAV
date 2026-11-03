import numpy as np
import cv2 
import matplotlib.pyplot as plt
import net_classifier as nc   # no circular import: net_classifier imports lane_pipeline only inside bev_data()

X_RANGE = (-7.0, 7.0)
Z_RANGE = (7.0, 20.0)
PX_PER_M = 30
GAUSS_KSIZE = 5

DARK_GRAD_THR = 60     # |Sobel x| threshold for the gray->black / black->gray transitions
DARK_MAX_W_PX = 9      # max width of a dark line in BEV pixels (9 px = 0.3 m at 30 px/m)
DARK_V_MAX = 70        # the pixel between the two edges must be darker than this (0-255)
DARK_DILATE_PX = 3     # grow the dark mask to also cover the Canny edges next to it
DARK_BH_KSIZE = 15     # horizontal black-hat kernel: must be wider than the widest dark line (DARK_MAX_W_PX)
DARK_BH_THR = 18       # how much darker than the surroundings (0-255) a pixel must be
DARK_BH_MIN_LEN_PX = 25  # vertical opening: keeps only structures at least this long
DARK_EDGE_NEAR_PX = 8   # dark pixels closer than this to the road mask border are NOT removed (they are real edges)

EGO_MIN_AREA_PX = 3000     # smaller ego mask -> considered missing
EGO_IN_ROAD_MIN = 0.9      # fraction of the ego mask that must lie inside the road mask

def read_calib(path):
    calib = {}
    with open(path) as f:
        for row in f:
            if ":" in row:
                name, values = row.split(":", 1)
                calib[name.strip()] = np.array(values.split(), dtype=float)
    return calib

def compute_trapezoid(calib, x_range=X_RANGE, z_range=Z_RANGE):
    P2 = calib["P2"].reshape(3, 4)
    R0_rect = calib["R0_rect"].reshape(3, 3)
    T = calib["Tr_cam_to_road"].reshape(3, 4)
    R, t = T[:, :3], T[:, 3]

    def show(x, z):
        p_road = np.array([x, 0.0, z])           
        p_cam = R.T @ (p_road - t)               
        p_cam_rect = R0_rect @ p_cam                
        u, v, w = P2 @ np.append(p_cam_rect, 1.0)   
        return [u / w, v / w]                      

    return np.float32([
        show(x_range[0], z_range[0]),          
        show(x_range[1], z_range[0]),          
        show(x_range[1], z_range[1]),          
        show(x_range[0], z_range[1]),          
    ])

def apply_roi(img, trapezoid):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [trapezoid.astype(np.int32)], 255)   
    return cv2.bitwise_and(img, img, mask=mask)            

def brid_eye_view(img, trapezoid, x_range=X_RANGE, z_range=Z_RANGE, px_per_m=PX_PER_M):
    width = int((x_range[1] - x_range[0]) * px_per_m)
    height = int((z_range[1] - z_range[0]) * px_per_m)
    trapezoid_dst = np.float32([[0, height], [width, height], [width, 0], [0, 0]])  
    M = cv2.getPerspectiveTransform(trapezoid, trapezoid_dst)      
    M_inv = cv2.getPerspectiveTransform(trapezoid_dst, trapezoid)  
    bew = cv2.warpPerspective(img, M, (width, height)) 
    return bew, M, M_inv

def dark_line_mask(img_bev, grad_thr=DARK_GRAD_THR, max_w=DARK_MAX_W_PX, v_max=DARK_V_MAX, dilate_px=DARK_DILATE_PX):
    """
    Thin dark lines (gray -> black -> gray) found with the horizontal gradient: a strong negative
    Sobel x (entering the dark band) followed, at most max_w px to the right, by a strong positive one
    (leaving it), with a really dark pixel in between. Returns a uint8 0/255 mask.
    """
    gray = cv2.cvtColor(img_bev, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    falling, rising = gx < -grad_thr, gx > grad_thr
    W = gray.shape[1]

    mask = np.zeros(gray.shape, bool)
    for k in range(2, max_w + 1):
        center = blur[:, k // 2: k // 2 + W - k]
        pair = falling[:, :W - k] & rising[:, k:] & (center < v_max)
        mask[:, :W - k] |= pair      # left edge of the dark line
        mask[:, k:] |= pair          # right edge of the dark line
    out = mask.astype(np.uint8) * 255

    # relative criterion: thin lines darker than their horizontal surroundings (low-contrast lines too)
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (DARK_BH_KSIZE, 1))
    bh = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, bh_kernel)
    rel = (bh > DARK_BH_THR).astype(np.uint8) * 255
    rel = cv2.morphologyEx(rel, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, DARK_BH_MIN_LEN_PX)))
    out = cv2.bitwise_or(out, rel)

    if dilate_px > 0:
        out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2))
    return out

def valid_bev_mask(img_shape, M, size):
    """BEV pixels that come from inside the original image (True) vs. outside (False)."""
    ones = np.full(img_shape[:2], 255, np.uint8)
    return cv2.warpPerspective(ones, M, size, flags=cv2.INTER_NEAREST) > 0


def ego_mask_bev(img_bird_eye, valid_bev):
    """Ego-lane mask (uint8 0/255) from the 2nd U-Net channel. Empty mask if the net finds nothing."""
    h, w = valid_bev.shape
    p = nc.ego_prob_bev(img_bird_eye, valid_bev)
    mask = (p > nc.ROAD_THR).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return np.zeros_like(mask)
    seed = lab[h - 10, w // 2]
    if seed == 0:                                # ego lane must lie under the car: otherwise the net is wrong
        return np.zeros_like(mask)
    keep = seed
    mask = ((lab == keep) * 255).astype(np.uint8)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, cnts, -1, 255, cv2.FILLED)
    filled = cv2.medianBlur(filled, 15)
    return filled if (filled > 0).sum() >= EGO_MIN_AREA_PX else np.zeros_like(mask)

def pick_guide_mask(ego_mask, road_mask):
    """Ego-lane mask if present and consistent with the road, otherwise the road mask."""
    if ego_mask.any():
        inside = ((ego_mask > 0) & (road_mask > 0)).sum() / (ego_mask > 0).sum()
        if inside >= EGO_IN_ROAD_MIN:
            return dict(mask=ego_mask, source="lane")
    if road_mask.any():
        return dict(mask=road_mask, source="road")
    return dict(mask=None, source=None)

# ---------------- Nuova pipeline ispirata al GitHub ----------------
def canny_bev_pipeline(img_bev):
    gray = cv2.cvtColor(img_bev, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (GAUSS_KSIZE, GAUSS_KSIZE), 0)
    canny_out = cv2.Canny(blur, 50, 150)

    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.uint8(np.absolute(sobelx))
    _, vert_mask = cv2.threshold(abs_sobelx, 30, 255, cv2.THRESH_BINARY)

    edges = cv2.bitwise_and(canny_out, vert_mask)
    return dict(gray=gray, clahe=clahe, blur=blur, canny=canny_out, vert_mask=vert_mask, edges=edges)


def lane_pipeline(path, img, out_path=None):
    calib = read_calib(path)
    trapezoid = compute_trapezoid(calib)
    img_roi = apply_roi(img, trapezoid)
    img_bird_eye, M, M_inv = brid_eye_view(img_roi, trapezoid)
    size = (img_bird_eye.shape[1], img_bird_eye.shape[0])
    valid_bev = valid_bev_mask(img.shape, M, size)

    steps = canny_bev_pipeline(img_bird_eye)

    # 1) road mask from the U-Net: keep only the edges inside the (eroded) road
        # 1) road mask from the U-Net: keep only the edges inside the road
    road_mask = nc.road_mask_bev(img_bird_eye, valid_bev)
    road_keep = road_mask
    if not road_keep.any():                      # net failed: do not wipe out every edge
        print("warning: empty road mask, road filter skipped")
        road_keep = np.full_like(road_mask, 255)

    guide = pick_guide_mask(ego_mask_bev(img_bird_eye, valid_bev), road_mask)
    guide["bev"] = img_bird_eye

    # 2) black-line mask: those pixels cannot become RANSAC points
    dark = dark_line_mask(img_bird_eye)
    if road_mask.any():                          # keep dark pixels near the road border: they are the road edge itself
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DARK_EDGE_NEAR_PX + 1,) * 2)
        near_edge = cv2.dilate(nc.road_edge(road_mask, valid_bev), k)
        dark[near_edge > 0] = 0

    img_thresh = cv2.bitwise_and(steps["edges"], road_keep)
    img_thresh = cv2.bitwise_and(img_thresh, cv2.bitwise_not(dark))

    steps.update(road_keep=road_keep, dark=dark, final=img_thresh)
    plot_pipeline_steps(img, img_bird_eye, trapezoid, steps, out_path)
    return M_inv, img_thresh, guide

def plot_pipeline_steps(img, img_bird_eye, trapezoid, steps, out_path=None):
    vis = img.copy()
    cv2.polylines(vis, [trapezoid.astype("int32")], True, (255, 0, 0), 3)

    road_ov = img_bird_eye.copy()
    road_ov[steps["road_keep"] > 0] = (0.6 * road_ov[steps["road_keep"] > 0] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)

    panels = [
        ("1. Original + ROI", vis, None),
        ("2. Bird-eye", img_bird_eye, None),
        ("3. Gray + CLAHE", steps["clahe"], "gray"),
        ("4. Gaussian blur", steps["blur"], "gray"),
        ("5. Canny", steps["canny"], "gray"),
        ("6. Sobel-x mask", steps["vert_mask"], "gray"),
        ("7. Canny AND Sobel-x", steps["edges"], "gray"),
        ("8. U-Net road mask", road_ov, None),
        ("9. Dark-line mask", steps["dark"], "gray"),
        ("10. Final (road AND NOT dark)", steps["final"], "gray"),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    for a, (title, im, cmap) in zip(axes.ravel(), panels):
        a.imshow(im, cmap=cmap)
        a.set_title(title)
        a.axis("off")
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
    else:
        plt.show()