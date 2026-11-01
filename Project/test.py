import copy, time
import numpy as np
import torch
import net_classifier as nc

H, W = 384, 640   # set to the real BEV shape
BATCH = 8
dev = torch.device("xpu")

torch.manual_seed(0)
net_cpu = nc.UNetR18(pretrained=False)
net_xpu = copy.deepcopy(net_cpu).to(dev)

x = torch.randn(BATCH, 3, H, W)
y = (torch.rand(BATCH, 2, H, W) > 0.5).float()
m = torch.ones(BATCH, 2, H, W)
x_d, y_d, m_d = x.to(dev), y.to(dev), m.to(dev)

def step(net, x, y, m):
    net.zero_grad()
    logits = net(x)
    loss = (nc.masked_dice_bce(logits[:, :1], y[:, :1], m[:, :1])
            + nc.masked_dice_bce(logits[:, 1:], y[:, 1:], m[:, 1:]))
    loss.backward()
    return loss.item()   # .item() forces the sync

# 1) correctness: same weights, same data, one forward+backward
l_cpu = step(net_cpu, x, y, m)
l_xpu = step(net_xpu, x_d, y_d, m_d)
print(f"loss cpu {l_cpu:.6f} | xpu {l_xpu:.6f}")
grads = [(n, p_c.grad, p_x.grad.cpu())
         for (n, p_c), (_, p_x) in zip(net_cpu.named_parameters(), net_xpu.named_parameters())]
scale = max(g_c.norm().item() for _, g_c, _ in grads)
worst, skipped = (0.0, ""), 0
for name, g_c, g_x in grads:
    if g_c.norm().item() < 1e-4 * scale:   # e.g. conv bias before BatchNorm: gradient is pure noise
        skipped += 1
        continue
    rel = ((g_c - g_x).norm() / g_c.norm()).item()
    if rel > worst[0]:
        worst = (rel, name)
print(f"worst grad rel.err: {worst[0]:.2e} at {worst[1]} ({skipped} tiny-grad params skipped)")

# 1b) fp64 CPU reference: is the XPU error just fp32 noise?
net_ref = copy.deepcopy(net_cpu).double()
step(net_ref, x.double(), y.double(), m.double())
rows = []
for (n, p_r), (_, p_c), (_, p_x) in zip(net_ref.named_parameters(),
                                         net_cpu.named_parameters(),
                                         net_xpu.named_parameters()):
    g_r = p_r.grad
    if g_r.norm().item() < 1e-4 * scale:
        continue
    e_c = ((p_c.grad.double() - g_r).norm() / g_r.norm()).item()
    e_x = ((p_x.grad.cpu().double() - g_r).norm() / g_r.norm()).item()
    rows.append((e_x, e_c, n))
rows.sort(reverse=True)
for e_x, e_c, n in rows[:5]:
    print(f"{n:28s} xpu-vs-fp64 {e_x:.2e} | cpu-vs-fp64 {e_c:.2e}")

# 2) speed of one training step
def bench(net, x, y, m, iters=3):
    for _ in range(2):
        step(net, x, y, m)   # warmup
    t = time.perf_counter()
    for _ in range(iters):
        step(net, x, y, m)
    return (time.perf_counter() - t) / iters

t_cpu = bench(net_cpu, x, y, m)
t_xpu = bench(net_xpu, x_d, y_d, m_d)
print(f"step: cpu {t_cpu*1e3:.0f} ms | xpu {t_xpu*1e3:.0f} ms | speedup {t_cpu/t_xpu:.2f}x")

# 3) CPU augmentation cost per batch (what the GPU cannot speed up)
rng = np.random.default_rng(0)
img = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
t = time.perf_counter()
for _ in range(10):
    nc.add_shadows(img, rng)
t_aug = (time.perf_counter() - t) / 10 * BATCH * 0.7
print(f"augmentation per batch (CPU): {t_aug*1e3:.0f} ms")