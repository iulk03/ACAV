# road_net.py
"""
Pixel-wise road / non-road segmentation in the bird-eye view with a small U-Net (ResNet18 encoder).
Output saved in road_net.pt file.
"""
import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR / "data_road"
NET_PATH = BASE_DIR / "road_net.pt"

ROAD_THR = 0.5       # threshold on the road probability
CHROMA_MIN = 40.0    # colour gradient threshold (OpenCV Lab, Sobel 3x3)
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ---------------- ground truth and data (copied from road_classifier.py) ----------------
def road_gt(image_path):
    """(road, valid) boolean masks from the KITTI Road GT. Name: um_000093 -> um_road_000093."""
    p = Path(image_path)
    prefix, num = p.stem.split("_")
    gt_path = p.parents[1] / "gt_image_2" / f"{prefix}_road_{num}.png"
    gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
    # OpenCV reads BGR: channel 0 (blue) = road, channel 2 (red) = valid GT area
    return gt[:, :, 0] > 0, gt[:, :, 2] > 0


def lane_gt(image_path):
    """(ego_lane, valid) boolean masks from um_lane_XXXXXX.png, or None if the image has no lane GT (umm, uu)."""
    p = Path(image_path)
    prefix, num = p.stem.split("_")
    gt_path = p.parents[1] / "gt_image_2" / f"{prefix}_lane_{num}.png"
    if not gt_path.exists():
        return None
    gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
    return gt[:, :, 0] > 0, gt[:, :, 2] > 0


def list_images():
    return sorted(glob.glob(str(ROOT / "training" / "image_2" / "*.png")))


def bev_data(image_path):
    """BEV image + road GT + valid area + ego-lane GT + lane valid area, warped like the pipeline."""
    import lane_pipeline as lp   # local import: avoids the circular import lane_pipeline -> road_net

    img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    road, valid = road_gt(image_path)
    lane = lane_gt(image_path)
    if lane is None:
        lane, lane_valid = np.zeros_like(road), np.zeros_like(valid)
    else:
        lane, lane_valid = lane
    calib = lp.read_calib(ROOT / "training" / "calib" / f"{Path(image_path).stem}.txt")
    trapezoid = lp.compute_trapezoid(calib)
    img_bev, M, _ = lp.brid_eye_view(lp.apply_roi(img, trapezoid), trapezoid)
    size = (img_bev.shape[1], img_bev.shape[0])

    def warp(m):
        return cv2.warpPerspective(m.astype(np.uint8) * 255, M, size, flags=cv2.INTER_NEAREST) > 0

    return img_bev, warp(road), warp(valid), warp(lane), warp(lane_valid)


def add_shadows(img, rng):
    """Synthetic shadows / sun patches (image only, the GT does not change): data augmentation."""
    h, w = img.shape[:2]
    noise = 0.
    for cell in (20, 60):                                   # small patches + large shadow areas
        n = rng.random((h // cell + 2, w // cell + 2)).astype(np.float32)
        noise = noise + cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise / 2, (0, 0), 3)
    thr = np.quantile(noise, rng.uniform(0.3, 0.7))
    shade = np.clip((noise - thr) / rng.uniform(0.03, 0.08) * 0.5 + 0.5, 0, 1)[..., None]
    dark = rng.uniform(0.35, 0.7)                           # how dark the shadow is
    tint = np.array([0.92, 0.97, 1.05], np.float32)         # shadows are slightly bluer
    out = img.astype(np.float32) * (1 - dark * shade) * (1 + (tint - 1) * shade)
    out *= rng.uniform(0.85, 1.15)                          # global exposure
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------- network ----------------
class UNetR18(nn.Module):
    """ResNet18 encoder + light decoder with skip connections. Output: 1 road logit per pixel."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        r = torchvision.models.resnet18(weights=weights)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)                              # /2,  64 ch
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4    # /4 64, /8 128, /16 256, /32 512
        self.d4 = self._block(512 + 256, 256)
        self.d3 = self._block(256 + 128, 128)
        self.d2 = self._block(128 + 64, 64)
        self.d1 = self._block(64 + 64, 32)
        self.head = nn.Conv2d(32, 2, 1)

    @staticmethod
    def _block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    @staticmethod
    def _up(x, ref):
        return nn.functional.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.l1(self.pool(s0))
        s2 = self.l2(s1)
        s3 = self.l3(s2)
        s4 = self.l4(s3)
        d = self.d4(torch.cat([self._up(s4, s3), s3], 1))
        d = self.d3(torch.cat([self._up(d, s2), s2], 1))
        d = self.d2(torch.cat([self._up(d, s1), s1], 1))
        d = self.d1(torch.cat([self._up(d, s0), s0], 1))
        return self._up(self.head(d), x)


def to_tensor(bev_rgb):
    """HxWx3 uint8 RGB -> 1x3xHxW normalised float tensor."""
    x = (bev_rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


def masked_dice_bce(logits, target, valid):
    """BCE + Dice on the valid pixels. A sample with an all-False mask contributes zero loss."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logits) * valid
    t = target * valid
    inter = (p * t).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


def fmt_time(seconds):
    s = int(seconds)
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


@torch.no_grad()
def eval_road_iou(net, val_data):
    """Mean road IoU on the validation images (original, no shadows): cheap check of the training state."""
    net.eval()
    ious = []
    for _, (bev, road, valid, _, _) in val_data:
        p = torch.sigmoid(net(to_tensor(bev).to(DEVICE))[0, 0]).cpu().numpy() > ROAD_THR
        ious.append((p & road & valid).sum() / max(((p | road) & valid).sum(), 1))
    return float(np.mean(ious))


# ---------------- training ----------------
def train(epochs=60, batch=8, val_frac=0.2, lr=3e-4):
    files = list_images()
    rng = np.random.default_rng(SEED)
    rng.shuffle(files)
    n_val = int(len(files) * val_frac)
    val_files, train_files = files[:n_val], files[n_val:]          # split per IMAGE, not per pixel
    torch.manual_seed(SEED)

    train_data = []
    for k, f in enumerate(train_files, start=1):
        train_data.append(bev_data(f))
        print(f"\rloading train data {k}/{len(train_files)}", end="", flush=True)
    print()
    val_data = [(Path(f).stem.split("_")[0], bev_data(f)) for f in val_files]
    n_lane = sum(int(d[4].any()) for d in train_data)
    print(f"train: {len(train_data)} images ({n_lane} with ego-lane GT), val: {len(val_data)}, device: {DEVICE}")

    net = UNetR18().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def tensor(a):
        return torch.from_numpy(np.ascontiguousarray(a)).float()[None, None]

    EVAL_EVERY = 5
    history = {"loss": [], "val_epochs": [], "val_iou": []}
    t0 = time.time()

    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(train_data))
        # batches with a single sample are skipped (BatchNorm), so count only the real ones
        starts = [s for s in range(0, len(order), batch) if min(batch, len(order) - s) >= 2]
        n_batches = len(starts)
        total = 0.0

        for b, s in enumerate(starts, start=1):
            xs, ys, ms = [], [], []
            for i in order[s:s + batch]:
                bev, road, valid, lane, lane_valid = train_data[i]
                if rng.random() < 0.7:
                    bev = add_shadows(bev, rng)
                if rng.random() < 0.5:                             # horizontal flip
                    bev, road, valid, lane, lane_valid = (a[:, ::-1] for a in (bev, road, valid, lane, lane_valid))
                xs.append(to_tensor(np.ascontiguousarray(bev)))
                ys.append(torch.cat([tensor(road), tensor(lane)], 1))
                ms.append(torch.cat([tensor(valid), tensor(lane_valid)], 1))
            x, y, m = (torch.cat(t).to(DEVICE) for t in (xs, ys, ms))
            logits = net(x)
            loss = masked_dice_bce(logits[:, :1], y[:, :1], m[:, :1]) + \
                   masked_dice_bce(logits[:, 1:], y[:, 1:], m[:, 1:])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

            # live progress line (overwritten in place)
            done = ep * n_batches + b
            all_batches = epochs * n_batches
            elapsed = time.time() - t0
            eta = elapsed / done * (all_batches - done)
            filled = int(30 * done / all_batches)
            print(f"\r[{'#' * filled}{'-' * (30 - filled)}] {100 * done / all_batches:5.1f}% | "
                  f"epoch {ep + 1}/{epochs} | batch {b}/{n_batches} | loss {total / b:.4f} | "
                  f"elapsed {fmt_time(elapsed)} | ETA {fmt_time(eta)}", end="", flush=True)

        sched.step()
        history["loss"].append(total / max(n_batches, 1))
        msg = f"\nepoch {ep + 1}/{epochs} done: loss {history['loss'][-1]:.4f}"
        if (ep + 1) % EVAL_EVERY == 0 or ep == epochs - 1:
            iou = eval_road_iou(net, val_data)
            history["val_epochs"].append(ep + 1)
            history["val_iou"].append(iou)
            msg += f" | val road IoU {iou:.3f}"
        print(msg)

    print(f"training finished in {fmt_time(time.time() - t0)}")
    torch.save(net.state_dict(), NET_PATH)
    print("saved:", NET_PATH)

    # training curve: loss per epoch (left axis) and validation IoU (right axis)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(history["loss"]) + 1), history["loss"], color="tab:blue", label="train loss")
    ax.set(xlabel="epoch", ylabel="train loss")
    ax2 = ax.twinx()
    ax2.plot(history["val_epochs"], history["val_iou"], "o-", color="tab:red", label="val road IoU")
    ax2.set_ylabel("val road IoU")
    fig.tight_layout()
    fig.savefig(BASE_DIR / "training_curve.png", dpi=150)
    plt.close(fig)

    # validation IoU: road per category (um/umm/uu), ego-lane on the images that have it
    net.eval()
    val_rng = np.random.default_rng(123)
    ious = {}
    with torch.no_grad():
        for key, (bev, road, valid, lane, lane_valid) in val_data:
            for tag, im in (("orig", bev), ("shadow", add_shadows(bev, val_rng))):
                p = torch.sigmoid(net(to_tensor(im).to(DEVICE))[0]).cpu().numpy() > ROAD_THR
                inter = (p[0] & road & valid).sum()
                union = ((p[0] | road) & valid).sum()
                ious.setdefault((f"road {key}", tag), []).append(inter / max(union, 1))
                if lane_valid.any():
                    inter = (p[1] & lane & lane_valid).sum()
                    union = ((p[1] | lane) & lane_valid).sum()
                    ious.setdefault(("ego-lane", tag), []).append(inter / max(union, 1))
    for (k, tag), v in sorted(ious.items()):
        print(f"val IoU [{k}, {tag}]: {np.mean(v):.3f}  ({len(v)} img)")
    return net



# ---------------- inference in the bird-eye view ----------------
_net = None


def load_net():
    global _net
    if _net is None:
        _net = UNetR18(pretrained=False).to(DEVICE)
        _net.load_state_dict(torch.load(NET_PATH, map_location=DEVICE))
        _net.eval()
    return _net


@torch.no_grad()
def road_prob_bev(img_bird_eye, valid_bev, net=None):
    """Road probability for every BEV pixel (useful for diagnosis)."""
    net = net or load_net()
    p = torch.sigmoid(net(to_tensor(img_bird_eye).to(DEVICE))[0, 0]).cpu().numpy()
    p[~valid_bev] = 0
    return p

@torch.no_grad()
def ego_prob_bev(img_bird_eye, valid_bev, net=None):
    """Ego-lane probability for every BEV pixel."""
    net = net or load_net()
    p = torch.sigmoid(net(to_tensor(img_bird_eye).to(DEVICE))[0, 1]).cpu().numpy()
    p[~valid_bev] = 0
    return p


def road_mask_bev(img_bird_eye, valid_bev, net=None):
    """Road mask (uint8 0/255) in the BEV: threshold, cleaning, ego component, filled holes, smooth border."""
    h, w = valid_bev.shape
    mask = (road_prob_bev(img_bird_eye, valid_bev, net) > ROAD_THR).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)     # removes isolated blobs (bright grass, etc.)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)    # closes small holes

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        seed = lab[h - 10, w // 2]                       # pixel in front of the car
        keep = seed if seed > 0 else 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = ((lab == keep) * 255).astype(np.uint8)

    # the road has no holes: shadows, cracks and inner patches must not generate contours
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, cnts, -1, 255, cv2.FILLED)
    return cv2.medianBlur(filled, 15)                    # less jagged border


# ---------------- edges (copied from road_classifier.py) ----------------
def road_edge(mask, valid_bev, margin=8):
    """Contour of the road mask, without the fake border caused by the black areas outside the image."""
    edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    safe = cv2.erode(valid_bev.astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    edge[safe == 0] = 0
    return edge


def chroma_edges(img_bev, valid_bev, min_grad=CHROMA_MIN, percentile=97, block=60, margin=8):
    """
    COLOUR edges (not luminance), almost vertical: asphalt/grass border even when the luminance is similar;
    shadow edges (which mostly change luminance) stay under the threshold.
    """
    ab = cv2.cvtColor(cv2.GaussianBlur(img_bev, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32)[:, :, 1:]
    gx = np.sqrt(sum(cv2.Sobel(ab[:, :, c], cv2.CV_32F, 1, 0, ksize=3) ** 2 for c in range(2)))
    gy = np.sqrt(sum(cv2.Sobel(ab[:, :, c], cv2.CV_32F, 0, 1, ksize=3) ** 2 for c in range(2)))
    g = np.hypot(gx, gy)

    safe = cv2.erode(valid_bev.astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)) > 0
    out = np.zeros(g.shape, np.uint8)
    for c in range(0, g.shape[1], block):
        ok = safe[:, c:c + block]
        if ok.any():
            thr = max(np.percentile(g[:, c:c + block][ok], percentile), min_grad)
            out[:, c:c + block] = (g[:, c:c + block] > thr) * 255
    out[~safe | (gx < 1.5 * gy)] = 0          # only almost vertical edges (horizontal gradient)
    return out


if __name__ == "__main__":
    import lane_fitting  # (import order: lane_fitting first, avoids the circular import)
    train()