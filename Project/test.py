import time, torch, torch_directml

dml = torch_directml.device()
print(torch_directml.device_name(0))

N = 2048
a, b = torch.randn(N, N), torch.randn(N, N)   # float32 di default
ref = a @ b                                    # riferimento CPU

def bench(dev, iters=10):
    x, y = a.to(dev), b.to(dev)
    for _ in range(3):                         # warmup (compilazione kernel)
        _ = x @ y
    _ = (x @ y).cpu()                          # sync
    t = time.perf_counter()
    for _ in range(iters):
        z = x @ y
    out = z.cpu()                              # .cpu() forza la sync, altrimenti misuri solo il lancio asincrono
    return out, (time.perf_counter() - t) / iters

out_dml, t_dml = bench(dml)
_, t_cpu = bench("cpu")

rel_err = (out_dml - ref).abs().max() / ref.abs().max()
print(f"DML {t_dml*1e3:.1f} ms | CPU {t_cpu*1e3:.1f} ms | rel.err {rel_err:.2e}")