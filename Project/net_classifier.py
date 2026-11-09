"""
Pixel-wise road / non-road segmentation in the bird-eye view with a small U-Net (ResNet18 encoder).
Output saved in road_net.pt file.
"""
import glob
from pathlib import Path
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import time
import matplotlib
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR / "data_road"
NET_PATH = BASE_DIR / "road_net.pt"

ROAD_THR = 0.5       # threshold on the road probability
CHROMA_MIN = 40.0    # colour gradient threshold (OpenCV Lab, Sobel 3x3)
BDD_DIR = Path("/kaggle/input/datasets/solesensei/solesensei_bdd100k")
EGO_PATH = Path("ego_net.pt")                    # cwd = /kaggle/working, shows up in the Output tab
IMG_SIZE = (416, 128)      # (W, H) network input, KITTI-like aspect (3.25)
CROP = (230, 618)          # BDD rows with a KITTI-like field of view (to be checked visually)
SEED = 0
import torch

def pick_device():
    import os
    forced = os.environ.get("FORCE_DEVICE")          # e.g. FORCE_DEVICE=cpu to bypass any backend
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():                    # 1) CUDA always has priority
        return torch.device("cuda")
    try:                                             # 2) DirectML only if CUDA is not available
        import torch_directml
        if torch_directml.device_count() > 0:
            return torch_directml.device()
    except ImportError:
        pass
    if hasattr(torch, "xpu") and torch.xpu.is_available():   # 3) Intel XPU
        return torch.device("xpu")
    return torch.device("cpu")

INFER_DEVICE = torch.device("cpu") 
DEVICE = pick_device()
print("Device:", DEVICE)

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
    bce = logits.clamp(min=0) - logits * target + torch.log1p(torch.exp(-logits.abs()))
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
def eval_road_iou(net, val_data, device):
    net.eval()
    ious = []
    for _, (bev, road, valid, _, _) in val_data:
        p = torch.sigmoid(net(to_tensor(bev).to(device))[0, 0]).cpu().numpy() > ROAD_THR
        ious.append((p & road & valid).sum() / max(((p | road) & valid).sum(), 1))
    return float(np.mean(ious))


# ---------------- training ----------------
def train(epochs=60, batch=8, val_frac=0.2, lr=3e-4):
    device = pick_device()
    print("Training device:", device)
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
    print(f"train: {len(train_data)} images ({n_lane} with ego-lane GT), val: {len(val_data)}, device: {device}")

    net = UNetR18().to(device)
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
            x, y, m = (torch.cat(t).to(device) for t in (xs, ys, ms))
            logits = net(x)
            loss = masked_dice_bce(logits[:, :1], y[:, :1], m[:, :1]) + masked_dice_bce(logits[:, 1:], y[:, 1:], m[:, 1:])
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
            iou = eval_road_iou(net, val_data, device)
            history["val_epochs"].append(ep + 1)
            history["val_iou"].append(iou)
            msg += f" | val road IoU {iou:.3f}"
        print(msg)

    print(f"training finished in {fmt_time(time.time() - t0)}")
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, NET_PATH)
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
                p = torch.sigmoid(net(to_tensor(im).to(device))[0]).cpu().numpy() > ROAD_THR
                inter = (p[0] & road & valid).sum()
                union = ((p[0] | road) & valid).sum()
                ious.setdefault((f"road {key}", tag), []).append(inter / max(union, 1))
                if lane_valid.any():
                    inter = (p[1] & lane & lane_valid).sum()
                    union = ((p[1] | lane) & lane_valid).sum()
                    ious.setdefault(("ego-lane", tag), []).append(inter / max(union, 1))
    for (k, tag), v in sorted(ious.items()):
        print(f"val IoU [{k}, {tag}]: {np.mean(v):.3f}  ({len(v)} img)")
    return net.cpu()


def poly_points(poly, n=8):
    """BDD100K poly2d (vertices + types 'L'/'C') -> (M,2) polyline; cubic Bezier segments sampled with n points."""
    v, t = np.array(poly["vertices"], np.float32), poly["types"]
    N = len(v)
    L = [k for k, c in enumerate(t) if c == "L"]
    u = np.linspace(0, 1, n, endpoint=False)
    basis = np.stack([(1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u ** 2 * (1 - u), u ** 3], 1)   # (n,4)
    out = []
    for a, b in zip(L, L[1:] + [L[0] + N]):                      # last segment wraps around
        seg = v[[k % N for k in range(a, b + 1)]]
        out.append(basis @ seg if len(seg) == 4 else seg[:-1])   # 4 points = Bezier, 2 = straight
    return np.concatenate(out)


def drivable_mask(item):
    """BDD100K labels of one image -> (H,W) uint8 at IMG_SIZE: 0 direct, 1 alternative, 2 background.
    Crop and resize are applied to the vertices, so the full-res mask is never built."""
    W, H = IMG_SIZE
    scale = np.array([W / 1280, H / (CROP[1] - CROP[0])], np.float32)
    y = np.full((H, W), 2, np.uint8)
    labels = [l for l in item["labels"] if l["category"] == "drivable area"]
    for l in sorted(labels, key=lambda l: l["attributes"]["areaType"] == "direct"):   # alternative first, direct on top
        val = 0 if l["attributes"]["areaType"] == "direct" else 1
        for p in l["poly2d"]:
            pts = (poly_points(p) - np.array([0, CROP[0]], np.float32)) * scale
            cv2.fillPoly(y, [np.round(pts).astype(np.int32)], val)
    return y


def plot_history(history, out_path=None):
    """Train loss (left axis) and validation IoU for road / ego lane (right axis), one point per epoch."""
    ep = range(1, len(history["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep, history["loss"], color="tab:blue", label="train loss")
    ax.set(xlabel="epoch", ylabel="train loss")
    ax2 = ax.twinx()
    ax2.plot(ep, history["iou_road"], "o-", color="tab:red", label="val road IoU")
    ax2.plot(ep, history["iou_ego"], "s-", color="tab:green", label="val ego IoU")
    ax2.set(ylabel="val IoU", ylim=(0, 1))
    fig.legend(loc="lower left", bbox_to_anchor=(0.12, 0.12), fontsize=8)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig


def bdd_data(n=10000):
    """First n BDD100K val images with drivable labels, cropped + resized to IMG_SIZE -> uint8 x (N,3,H,W), y (N,1,H,W)."""
    root = BDD_DIR / "bdd100k" / "bdd100k" / "images" / "100k" / "val"
    with open(BDD_DIR / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_val.json") as f:
        items = json.load(f)
    xs, ys = [], []
    for it in items:
        y = drivable_mask(it)
        im = cv2.imread(str(root / it["name"]))
        if im is None or (y == 2).all():
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)[CROP[0]:CROP[1]]
        xs.append(cv2.resize(im, IMG_SIZE, interpolation=cv2.INTER_AREA))
        ys.append(y)
        if len(xs) % 200 == 0:
            print(f"\rloading BDD {len(xs)}/{n}", end="", flush=True)
        if len(xs) == n:
            break
    print()
    return torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).contiguous(), torch.from_numpy(np.stack(ys))[:, None]


def augment(x, y):
    """Batched GPU augmentation. x float (B,3,H,W) in [0,1], y (B,1,H,W). Shadows + flip, no per-image loops."""
    B, _, H, W = x.shape
    r = lambda *s: torch.rand(*s, device=x.device)
    noise = sum(nn.functional.interpolate(r(B, 1, H // c + 2, W // c + 2), size=(H, W), mode="bicubic") for c in (8, 24)) / 2
    idx = ((0.3 + 0.4 * r(B, 1)) * (H * W - 1)).long()                       # per-sample quantile
    thr = noise.flatten(1).sort(1).values.gather(1, idx).view(B, 1, 1, 1)
    shade = ((noise - thr) / (0.03 + 0.05 * r(B, 1, 1, 1)) * 0.5 + 0.5).clamp(0, 1)
    tint = torch.tensor([0.92, 0.97, 1.05], device=x.device).view(1, 3, 1, 1)
    dark = 0.35 + 0.35 * r(B, 1, 1, 1)
    xa = x * (1 - dark * shade) * (1 + (tint - 1) * shade) * (0.85 + 0.3 * r(B, 1, 1, 1))
    x = torch.where(r(B, 1, 1, 1) < 0.7, xa.clamp(0, 1), x)
    flip = r(B, 1, 1, 1) < 0.5
    return torch.where(flip, x.flip(3), x), torch.where(flip, y.flip(3), y)


def train_ego(epochs=25, batch=32, lr=3e-4, print_every=10, n_images=10000):
    """Train the U-Net on BDD100K in perspective: channel 0 = drivable (direct+alt), channel 1 = ego lane (direct)."""
    x, y = bdd_data(n_images)
    n_val = len(x) // 10
    n_train = len(x) - n_val
    torch.manual_seed(SEED)
    net = UNetR18().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mean, std = (torch.tensor(a, device=DEVICE).view(1, 3, 1, 1) for a in (MEAN, STD))
    target = lambda yb: torch.cat([yb < 2, yb == 0], 1).float()

    n_batches = sum(1 for s in range(0, n_train, batch) if min(batch, n_train - s) >= 2)
    t0 = time.time()

    for ep in range(epochs):
        net.train()
        ep_t0 = time.time()
        total, nb = 0.0, 0
        for b in torch.randperm(n_train).split(batch):
            if len(b) < 2:                                   # BatchNorm
                continue
            xb, yb = augment(x[n_val + b].to(DEVICE).float() / 255, y[n_val + b].to(DEVICE))
            t = target(yb)
            with torch.autocast("cuda", dtype=torch.float16):
                lg = net((xb - mean) / std).float()
            ones = torch.ones_like(t[:, :1])
            loss = masked_dice_bce(lg[:, :1], t[:, :1], ones) + masked_dice_bce(lg[:, 1:], t[:, 1:], ones)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total += loss.item()
            nb += 1
            if nb % print_every == 0 or nb == n_batches:
                done = ep * n_batches + nb
                all_batches = epochs * n_batches
                elapsed = time.time() - t0
                eta = elapsed / done * (all_batches - done)
                filled = int(30 * done / all_batches)
                print(f"\r[{'#' * filled}{'-' * (30 - filled)}] {100 * done / all_batches:5.1f}% | "
                      f"epoch {ep + 1}/{epochs} | batch {nb}/{n_batches} | loss {total / nb:.4f} | "
                      f"elapsed {fmt_time(elapsed)} | ETA {fmt_time(eta)}", end="", flush=True)
        sched.step()

        net.eval()
        with torch.no_grad():
            ious = []
            for xv, yv in zip(x[:n_val].split(64), y[:n_val].split(64)):
                p = torch.sigmoid(net((xv.to(DEVICE).float() / 255 - mean) / std)) > ROAD_THR
                t = target(yv.to(DEVICE)) > 0
                ious.append(((p & t).sum((2, 3)) / (p | t).sum((2, 3)).clamp(min=1)).mean(0))
            iou = torch.stack(ious).mean(0)
        print(f"\nepoch {ep + 1}/{epochs} done in {fmt_time(time.time() - ep_t0)}: "
              f"loss {total / max(nb, 1):.4f} | val IoU road {iou[0]:.3f} ego {iou[1]:.3f}")
        torch.save({k: v.cpu() for k, v in net.state_dict().items()}, EGO_PATH)   # checkpoint every epoch


@torch.no_grad()
def eval_ego_kitti(net_path=EGO_PATH):
    """Ego-lane IoU of a BDD-trained net on the KITTI images that have lane GT (um_*), perspective view."""
    net = load_net(net_path)
    ious = []
    for f in list_images():
        gt = lane_gt(f)
        if gt is None:
            continue
        lane, valid = gt
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        small = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
        p = torch.sigmoid(net(to_tensor(small).to(INFER_DEVICE))[0, 1]).numpy() > ROAD_THR
        lane_s = cv2.resize(lane.astype(np.uint8), IMG_SIZE, interpolation=cv2.INTER_NEAREST) > 0
        valid_s = cv2.resize(valid.astype(np.uint8), IMG_SIZE, interpolation=cv2.INTER_NEAREST) > 0
        ious.append((p & lane_s & valid_s).sum() / max(((p | lane_s) & valid_s).sum(), 1))
    print(f"KITTI ego IoU: {np.mean(ious):.3f} on {len(ious)} images")
    return float(np.mean(ious))


# ---------------- inference in the bird-eye view ----------------
_nets = {}

def load_net(path=NET_PATH):
    if path not in _nets:
        net = UNetR18(pretrained=False)
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        _nets[path] = net.to(INFER_DEVICE).eval()
    return _nets[path]

@torch.no_grad()
def road_prob_bev(img_bird_eye, valid_bev, net=None):
    net = net or load_net()
    p = torch.sigmoid(net(to_tensor(img_bird_eye).to(INFER_DEVICE))[0, 0]).numpy()
    p[~valid_bev] = 0
    return p

@torch.no_grad()
def ego_prob_bev(img, M, valid_bev):
    """Ego-lane probability: net on the ORIGINAL image (perspective, IMG_SIZE), then warped into the BEV."""
    h, w = img.shape[:2]
    small = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    p = torch.sigmoid(load_net(EGO_PATH)(to_tensor(small).to(INFER_DEVICE))[0, 1]).numpy()
    p = cv2.warpPerspective(cv2.resize(p, (w, h)), M, valid_bev.shape[::-1])
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
    import matplotlib
    matplotlib.use("Agg")   # headless only when run as a script
    train_ego()
