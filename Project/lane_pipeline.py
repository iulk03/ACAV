import numpy as np
import cv2 
import matplotlib.pyplot as plt
import net_classifier as nc   # no circular import: net_classifier imports lane_pipeline only inside bev_data()

X_RANGE = (-7.0, 7.0)
Z_RANGE = (7.0, 20.0)
PX_PER_M = 30
GAUSS_KSIZE = 5

EGO_MIN_AREA_PX = 3000      # smaller ego mask -> considered missing
EGO_IN_ROAD_MIN = 0.9       # fraction of the ego mask that must lie inside the road mask
ROAD_KEEP_PX = 6            # the road mask is widened by this many px before filtering the line centres (0.2 m at 30 px/m)

# ---- bright line detection (signed gradient on log-luminance) ----
LINE_W_PX = (4, 6, 8)       # allowed line widths (px)  ~0.13-0.27 m at 30 px/m
LINE_GRAD = 0.05            # threshold on the line response (log-luminance gradient per pixel)
LINE_MARGIN_PX = 10         # distance from the black area outside the original image


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


def valid_bev_mask(img_shape, M, size):
    """BEV pixels that come from inside the original image (True) vs. outside (False)."""
    ones = np.full(img_shape[:2], 255, np.uint8)
    return cv2.warpPerspective(ones, M, size, flags=cv2.INTER_NEAREST) > 0


def widen_mask(mask, valid_bev, px=1, margin=6):
    """Dilate a 0/255 mask by px pixels, without leaking into the black area outside the original image."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    out = cv2.dilate(mask, k)
    safe = cv2.erode(valid_bev.astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    out[safe == 0] = 0
    return out


def ego_mask_bev(img, M, valid_bev):
    """Ego-lane mask (uint8 0/255) from the 2nd U-Net channel. Empty mask if the net finds nothing."""
    h, w = valid_bev.shape
    p = nc.ego_prob_bev(img, M, valid_bev)
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


# ---------------- bright line detection: signed gradient ----------------
def _shift_x(a, s):
    """out[:, x] = a[:, x+s], zero outside the border."""
    out = np.zeros_like(a)
    if s > 0:
        out[:, :-s] = a[:, s:]
    elif s < 0:
        out[:, -s:] = a[:, :s]
    else:
        out[:] = a
    return out


def bright_line_response(gray, valid_bev):
    """Response for 'bright line': a rising edge on the left AND a falling edge on the right.
    Works on the log of the luminance -> relative contrast, independent of shadows.
      - white line (even in shadow): up then down  -> kept
      - dark crack / bitumen line:   down then up  -> discarded
      - shadow border:               single step   -> discarded
    """
    blur = cv2.GaussianBlur(gray, (GAUSS_KSIZE, GAUSS_KSIZE), 0).astype(np.float32)
    gx = cv2.Sobel(np.log(blur + 10.0), cv2.CV_32F, 1, 0, ksize=3) / 8.0   # /8 -> derivative per pixel
    rise, fall = np.maximum(gx, 0), np.maximum(-gx, 0)

    resp = np.zeros_like(gx)
    for w in LINE_W_PX:
        pair = np.minimum(_shift_x(rise, -(w // 2)), _shift_x(fall, w // 2))   # centred on the line centre
        resp = np.maximum(resp, pair)

    safe = cv2.erode(valid_bev.astype(np.uint8), np.ones((2 * LINE_MARGIN_PX + 1,) * 2, np.uint8)) > 0
    resp[~safe] = 0
    return blur, resp


def line_bev_pipeline(img_bev, valid_bev):
    gray = cv2.cvtColor(img_bev, cv2.COLOR_RGB2GRAY)
    blur, resp = bright_line_response(gray, valid_bev)

    # thinning: keep only the local maximum along x (= line centre), 1 px
    peak = (resp >= _shift_x(resp, 1)) & (resp >= _shift_x(resp, -1))
    edges = ((resp > LINE_GRAD) & peak).astype(np.uint8) * 255
    return dict(gray=gray, blur=blur, resp=resp, edges=edges)


def lane_pipeline(path, img, out_path=None):
    calib = read_calib(path)
    trapezoid = compute_trapezoid(calib)
    img_roi = apply_roi(img, trapezoid)
    img_bird_eye, M, M_inv = brid_eye_view(img_roi, trapezoid)
    size = (img_bird_eye.shape[1], img_bird_eye.shape[0])
    valid_bev = valid_bev_mask(img.shape, M, size)

    steps = line_bev_pipeline(img_bird_eye, valid_bev)
        
    # 1) road mask from the U-Net: keep only the edges inside the (widened) road
    road_mask = nc.road_mask_bev(img_bird_eye, valid_bev)
    if road_mask.any():
        road_keep = widen_mask(road_mask, valid_bev, px=ROAD_KEEP_PX)
    else:                                        # net failed: do not wipe out every edge
        print("warning: empty road mask, road filter skipped")
        road_keep = np.full_like(road_mask, 255)

    ego_mask = ego_mask_bev(img, M, valid_bev)
    w = (ego_mask > 0).sum(1) / PX_PER_M
    guide = pick_guide_mask(ego_mask, road_mask)
    guide["bev"] = img_bird_eye

    # 2) line centres inside the road: these pixels become the RANSAC points
    img_thresh = cv2.bitwise_and(steps["edges"], road_keep)

    steps.update(road_keep=road_keep, ego=ego_mask, final=img_thresh)
    if out_path is not None:
        plot_pipeline_steps(img, img_bird_eye, trapezoid, steps, out_path)
    return M_inv, img_thresh, guide

def plot_pipeline_steps(img, img_bird_eye, trapezoid, steps, out_path=None):
    vis = img.copy()
    cv2.polylines(vis, [trapezoid.astype("int32")], True, (255, 0, 0), 3)

    road_ov = img_bird_eye.copy()
    for key, color in (("road_keep", (0, 255, 0)), ("ego", (0, 191, 255))):   # RGB: green road, light blue ego
        m = steps[key] > 0
        road_ov[m] = (0.6 * road_ov[m] + 0.4 * np.array(color)).astype(np.uint8)
    panels = [
        ("1. Original + ROI", vis, None),
        ("2. Bird-eye", img_bird_eye, None),
        ("3. Gaussian blur", steps["blur"], "gray"),
        ("4. Line response", steps["resp"], "gray"),
        ("5. Line centres", steps["edges"], "gray"),
        ("6. U-Net road (green) + ego (blue)", road_ov, None),
        ("7. Final", steps["final"], "gray"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))        
    for a in axes.ravel():
        a.axis("off")
    for a, (title, im, cmap) in zip(axes.ravel(), panels):
        a.imshow(im, cmap=cmap)
        a.set_title(title)
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
    else:
        plt.show()