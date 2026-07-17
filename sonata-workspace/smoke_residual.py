#!/usr/bin/env python3
"""Fast synthetic smoke test for Stage-2 residual refinement (no dataset)."""
import os, sys, types
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

import training.train_residual_gan as TR
from models.point_cloud_vae import PointCloudVAE
from models.residual_refine import ResidualRefiner
from models.critic_token import TokenCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 512


def fake_loader(n=3, bsz=2):
    out = []
    for _ in range(n):
        cc, cb, pc, pb = [], [], [], []
        for b in range(bsz):
            nc = torch.randint(600, 1000, (1,)).item(); npp = torch.randint(300, 700, (1,)).item()
            cc.append(torch.randn(nc, 3)); cb.append(torch.full((nc,), b, dtype=torch.long))
            pc.append(torch.randn(npp, 3)); pb.append(torch.full((npp,), b, dtype=torch.long))
        out.append({"complete_coord": torch.cat(cc), "complete_batch": torch.cat(cb),
                    "partial_coord": torch.cat(pc), "partial_batch": torch.cat(pb)})
    return out


def args(**ov):
    a = types.SimpleNamespace(
        num_decoded_points=K, disc_warmup_epochs=0, lambda_adv=1.0,
        lambda_adv_ramp_epochs=0, conditional_critic=True, n_critic=2,
        lambda_gp=10.0, adaptive_adv=True, adaptive_max=1e4, adaptive_scale=0.2,
        gamma_offset=1.0, eps_drift=1e-3)
    for k, v in ov.items():
        setattr(a, k, v)
    return a


vae = PointCloudVAE(latent_dim=128, num_decoded_points=K, num_latent_tokens=8,
                    internal_dim=64, num_heads=4, num_dec_blocks=2).to(device).eval()
for p in vae.parameters():
    p.requires_grad_(False)
refiner = ResidualRefiner(dim=64, num_heads=4, num_blocks=2, max_offset=0.1).to(device)
disc = TokenCritic(in_dim=3, dim=64, num_tokens=8, num_heads=4, num_layers=1, conditional=True).to(device)
og = optim.AdamW(refiner.parameters(), lr=3e-4)
od = optim.Adam(disc.parameters(), lr=1e-4, betas=(0.0, 0.9))
w = SummaryWriter("/tmp/smoke_res_tb")
s = TR.train_epoch(vae, refiner, disc, og, od, fake_loader(), 0, args(), w, device)
w.close()
print(f"OK cd={s['cd']:.4g} off={s['off']:.3g} adv={s['adv']:.4g} aw={s['adapt_w']:.4g} cos={s['cos']}")
vcd = TR.validate(vae, refiner, fake_loader(2), device)
print(f"val OK val_cd={vcd:.4g}")
print("RESIDUAL SMOKE PASSED")
