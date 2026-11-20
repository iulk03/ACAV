import cv2
import numpy as np
import torch
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
import lane_fitting as lf
from lane_pipeline import X_RANGE, Z_RANGE, PX_PER_M

# ID ufficiali delle classi COCO (1-indexed in torchvision)
COCO_ROAD_CLASSES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}

# Dimensioni fisiche effettive (m) per costruire l'impronta a terra in BEV senza sovrastime
CLASS_LENGTH_M = {
    1: 0.6,   # person
    2: 1.7,   # bicycle
    3: 4.0,   # car
    4: 1.9,   # motorcycle
    6: 8.5,   # bus
    8: 7.5,   # truck
}

CLASS_WIDTH_M = {
    1: 0.6,   # person
    2: 0.7,   # bicycle
    3: 1.85,  # car
    4: 0.8,   # motorcycle
    6: 2.5,   # bus
    8: 2.5,   # truck
}

LANE_MARGIN_M = 0.05        # tolleranza laterale minima (m) sui bordi corsia
MAX_OBS_Z_M = 45.0          # distanza massima (m) oltre la quale ignorare gli ostacoli
BODY_LOWER_FRAC = 0.45      # frazione inferiore del bbox 2D (zona ruote/paraurti a contatto col suolo)


class ObstacleDetector:
    def __init__(self, conf=0.5, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf = conf
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
        self.transforms = weights.transforms()
        self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights).to(self.device)
        self.model.eval()
        self.target_ids = set(COCO_ROAD_CLASSES.keys())

    @torch.inference_mode()
    def detect(self, img_rgb):
        """Esegue inference COCO (torchvision) sull'immagine RGB."""
        # img_rgb: uint8 (H, W, 3) -> tensor float [0, 1] (3, H, W)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).to(self.device)
        inp = self.transforms(tensor)
        out = self.model([inp])[0]

        boxes = out["boxes"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        scores = out["scores"].cpu().numpy()

        detections = []
        for (x1, y1, x2, y2), cls_id, score in zip(boxes, labels, scores):
            cls_id = int(cls_id)
            if score < self.conf or cls_id not in self.target_ids:
                continue
            detections.append({
                "box": (float(x1), float(y1), float(x2), float(y2)),
                "cls_id": cls_id,
                "cls": COCO_ROAD_CLASSES[cls_id],
                "conf": float(score),
            })
        return detections


def image_to_bev_xz(uv_points, M):
    """Converte punti pixel (N, 2) dell'immagine originale in coordinate strada (x, z) in metri."""
    pts = np.asarray(uv_points, dtype=np.float32).reshape(-1, 1, 2)
    bev_px = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
    x = X_RANGE[0] + bev_px[:, 0] / PX_PER_M
    z = Z_RANGE[1] - bev_px[:, 1] / PX_PER_M
    return x, z


def _tight_ground_footprint(det, M, lat_margin_m=0.0):
    """
    Calcola i limiti reali a terra (x_min, x_max, z_front, z_back) riducendo il box
    all'effettiva larghezza e lunghezza del veicolo (senza sovrastime prospettiche).
    """
    x1, y1, x2, y2 = det["box"]
    xs, zs = image_to_bev_xz([(x1, y2), (x2, y2)], M)
    z_front = float(np.mean(zs))
    length_m = CLASS_LENGTH_M.get(det["cls_id"], 4.0)
    max_width_m = CLASS_WIDTH_M.get(det["cls_id"], 1.85)
    z_back = z_front + length_m

    x_lo = float(min(xs))
    x_hi = float(max(xs))
    raw_w = x_hi - x_lo
    x_c = 0.5 * (x_lo + x_hi)

    # Se il bbox 2D visto di scorcio proietta una larghezza maggiore dell'auto reale,
    # riduce l'impronta alla larghezza fisica effettiva ancorandola al lato esterno.
    if raw_w > max_width_m:
        if x_c > 0.5:
            x_lo = x_hi - max_width_m
        elif x_c < -0.5:
            x_hi = x_lo + max_width_m
        else:
            x_lo = x_c - 0.5 * max_width_m
            x_hi = x_c + 0.5 * max_width_m

    return x_lo - lat_margin_m, x_hi + lat_margin_m, z_front, z_back


def build_obstacle_bev_masks(detections, M, bev_shape, img_shape, pad_px=0, lat_margin_m=0.0):
    """
    Costruisce le rappresentazioni BEV ridotte all'effettiva area a terra degli ostacoli:
      - obs_body_bev   : uint8 (H_bev, W_bev), solo la fascia inferiore a contatto col suolo
                         warpata in BEV (evita di proiettare il tetto dell'auto sull'asfalto
                         cancellando le strisce della corsia e generando falsi negativi).
      - obs_ground_bev : uint8 (H_bev, W_bev), impronta fisica effettiva a terra.
      - obs_boxes      : lista di dict con limiti reali a terra (x_min, x_max, z_min, z_max, cls).
    """
    h_bev, w_bev = bev_shape
    h_img, w_img = img_shape[:2]
    obs_body_img = np.zeros((h_img, w_img), dtype=np.uint8)
    obs_ground_bev = np.zeros((h_bev, w_bev), dtype=np.uint8)
    obs_boxes = []

    if not detections:
        return np.zeros((h_bev, w_bev), dtype=np.uint8), obs_ground_bev, obs_boxes

    for det in detections:
        x1, y1, x2, y2 = det["box"]

        # 1) Maschera 2D ristretta alla sola fascia inferiore (ruote/paraurti), senza padding
        y_contact_top = y2 - BODY_LOWER_FRAC * (y2 - y1)
        ix1 = max(0, int(round(x1)) - pad_px)
        iy1 = max(0, int(round(y_contact_top)) - pad_px)
        ix2 = min(w_img, int(round(x2)) + pad_px)
        iy2 = min(h_img, int(round(y2)) + pad_px)
        if ix2 > ix1 and iy2 > iy1:
            obs_body_img[iy1:iy2, ix1:ix2] = 255

        # 2) Impronta fisica effettiva a terra (larghezza e lunghezza reali del veicolo)
        x_min, x_max, z_front, z_back = _tight_ground_footprint(det, M, lat_margin_m=lat_margin_m)

        if z_back < Z_RANGE[0] or z_front > Z_RANGE[1]:
            continue
        if x_max < X_RANGE[0] or x_min > X_RANGE[1]:
            continue

        z_min = max(Z_RANGE[0], z_front)
        z_max = min(Z_RANGE[1], z_back)
        if z_min >= z_max:
            continue

        obs_boxes.append(dict(
            x_min=x_min,
            x_max=x_max,
            z_min=z_min,
            z_max=z_max,
            cls=det["cls"],
        ))

        c1 = int(np.clip((x_min - X_RANGE[0]) * PX_PER_M, 0, w_bev - 1))
        c2 = int(np.clip((x_max - X_RANGE[0]) * PX_PER_M, 0, w_bev - 1))
        r1 = int(np.clip((Z_RANGE[1] - z_max) * PX_PER_M, 0, h_bev - 1))
        r2 = int(np.clip((Z_RANGE[1] - z_min) * PX_PER_M, 0, h_bev - 1))
        if c2 > c1 and r2 > r1:
            obs_ground_bev[r1:r2 + 1, c1:c2 + 1] = 255

    obs_body_bev = cv2.warpPerspective(obs_body_img, M, (w_bev, h_bev), flags=cv2.INTER_NEAREST)
    return obs_body_bev, obs_ground_bev, obs_boxes


def evaluate_obstacles(detections, M, curved=None, margin_m=LANE_MARGIN_M):
    """
    Calcola posizione (x, z), distanza dall'ego vehicle e verifica se l'ostacolo
    ricade effettivamente all'interno della corsia predetta (`curved`).
    """
    if not detections:
        return []

    feet_uv = [((d["box"][0] + d["box"][2]) * 0.5, d["box"][3]) for d in detections]
    xs, zs = image_to_bev_xz(feet_uv, M)

    enriched = []
    for det, x, z in zip(detections, xs, zs):
        if not (lf.EGO_Z_M < z <= MAX_OBS_Z_M):
            continue

        in_ego = False
        if curved is not None:
            z_arr = np.array([float(z)])
            x_left = float(lf.boundary_x(curved[0], z_arr)[0])
            x_right = float(lf.boundary_x(curved[1], z_arr)[0])
            in_ego = (x_left - margin_m) <= x <= (x_right + margin_m)

        dist_m = float(np.hypot(x, z - lf.EGO_Z_M))
        enriched.append({
            **det,
            "x": float(x),
            "z": float(z),
            "distance_m": dist_m,
            "in_ego_lane": in_ego,
        })

    return sorted(enriched, key=lambda o: o["distance_m"])


def draw_obstacles(img_rgb, obstacles):
    """
    Disegna i bounding box sull'immagine RGB:
      - Rosso spesso: ostacolo nella corsia ego
      - Giallo sottile: ostacolo fuori corsia
    """
    out = img_rgb.copy()
    for obs in obstacles:
        x1, y1, x2, y2 = map(int, obs["box"])
        in_lane = obs["in_ego_lane"]
        color = (255, 40, 40) if in_lane else (255, 200, 0)
        thickness = 3 if in_lane else 2

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        tag = "IN LANE " if in_lane else ""
        label = f"{tag}{obs['cls']} {obs['distance_m']:.1f}m"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_top = max(y1 - th - 6, 0)
        cv2.rectangle(out, (x1, y_top), (x1 + tw + 6, y_top + th + 6), color, -1)
        txt_color = (255, 255, 255) if in_lane else (0, 0, 0)
        cv2.putText(out, label, (x1 + 3, y_top + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, txt_color, 2, cv2.LINE_AA)
    return out