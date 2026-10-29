import numpy as np
import torch
from torchvision.models.detection import (fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCORE_MIN = 0.5
CLASS_IDS = {1, 2, 3, 4, 6, 8}      # COCO: person, bicycle, car, motorcycle, bus, truck

_det = None


def load_detector():
    global _det
    if _det is None:
        w = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        _det = fasterrcnn_mobilenet_v3_large_fpn(weights=w).to(DEVICE).eval()
    return _det


@torch.no_grad()
def detect_vehicles(img_rgb, score_min=SCORE_MIN):
    """COCO detections on the ORIGINAL image (the model normalises/resizes internally)."""
    x = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255).to(DEVICE)
    out = load_detector()([x])[0]
    return [dict(box=b.cpu().numpy(), label=int(l), score=float(s))
            for b, l, s in zip(out["boxes"], out["labels"], out["scores"])
            if s >= score_min and int(l) in CLASS_IDS]