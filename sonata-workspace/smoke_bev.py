#!/usr/bin/env python3
"""Fast synthetic smoke test for BEV-density residual refinement."""
import os, sys, types
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

import training.train_residual_bev as TB
from models.point_cloud_vae import PointCloudVAE
from models.residual_refine import ResidualRefiner
from models.bev_density import bev_jsd_loss, soft_bev_hist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 512

# --- unit check: bev_jsd_loss differentiable + zero on identical clouds ---
a = torch.randn(2000, 3, device=device, requires_grad=True)
same = bev_jsd_loss(a, a.detach())
g = torch.autograd.grad(bev_jsd_loss(a, torch.randn(2000, 3, device=device)), a)[0]
print(f"jsd(a,a)={same.item():.3e} (≈0), grad_norm_vs_other={g.norm().item():.3e} (>0)")


def fake_loader(n=3, bsz=3):
    out = []
    for _ in range(n):
        cc, cb = [], []
        for b in range(bsz):
            nc = torch.randint(600, 1000, (1,)).item()
            cc.append(torch.randn(nc, 3)); cb.append(torch.full((nc,), b, dtype=torch.long))
        out.append({"complete_coord": torch.cat(cc), "complete_batch": torch.cat(cb)})
    return out


args = types.SimpleNamespace(
    num_decoded_points=K, gamma_offset=1.0, lambda_bev=1.0, bev_grid=40, bev_sigma=0.05,
    bev_warmup_epochs=0, bev_ramp_epochs=0)

vae = PointCloudVAE(latent_dim=128, num_decoded_points=K, num_latent_tokens=8,
                    internal_dim=64, num_heads=4, num_dec_blocks=2).to(device).eval()
for p in vae.parameters():
    p.requires_grad_(False)
refiner = ResidualRefiner(dim=64, num_heads=4, num_blocks=2, max_offset=0.05).to(device)
og = optim.AdamW(refiner.parameters(), lr=3e-4)
w = SummaryWriter("/tmp/smoke_bev_tb")
s = TB.train_epoch(vae, refiner, og, fake_loader(), 0, args, w, device)
w.close()
print(f"train OK cd={s['cd']:.4g} off={s['off']:.3g} bev={s['bev']:.4g} cos={s['cos']}")
vcd, vbev = TB.validate(vae, refiner, fake_loader(2), args, device)
print(f"val OK val_cd={vcd:.4g} val_bev={vbev:.4g}")
print("BEV SMOKE PASSED")
