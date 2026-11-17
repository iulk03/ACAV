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

LANE_MARGIN_M = 0.25        # tolleranza laterale (m) oltre i bordi corsia
MAX_OBS_Z_M = 45.0          # distanza massima (m) oltre la quale ignorare gli ostacoli


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


def evaluate_obstacles(detections, M, curved=None, margin_m=LANE_MARGIN_M):
    """
    Calcola posizione (x, z), distanza dall'ego vehicle e verifica se l'ostacolo
    ricade all'interno della corsia predetta (`curved`).
    """
    if not detections:
        return []

    # Punto di contatto a terra = centro del lato inferiore del bounding box
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