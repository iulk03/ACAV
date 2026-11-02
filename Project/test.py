# smoke_dml.py (new file, same folder as net_classifier.py)
import time
import torch
import net_classifier as nc

print("device:", nc.DEVICE, "| IS_DML:", getattr(nc, "IS_DML", None))
net = nc.UNetR18(pretrained=False).to(nc.DEVICE)
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, foreach=False)
x = torch.randn(4, 3, 405, 420).to(nc.DEVICE)
y = (torch.rand(4, 2, 405, 420) > 0.5).float().to(nc.DEVICE)
m = torch.ones_like(y)

for i in range(3):
    t = time.time()
    logits = net(x)
    loss = nc.masked_dice_bce(logits[:, :1], y[:, :1], m[:, :1]) + nc.masked_dice_bce(logits[:, 1:], y[:, 1:], m[:, 1:])
    opt.zero_grad(); loss.backward(); opt.step()
    print(f"step {i}: loss {loss.item():.4f} | {time.time() - t:.2f}s")