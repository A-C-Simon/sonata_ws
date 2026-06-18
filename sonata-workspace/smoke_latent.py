#!/usr/bin/env python3
"""Fast synthetic smoke test for latent-space adversarial refinement."""
import os, sys, types
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

import training.train_latent_gan as TL
from models.point_cloud_vae import PointCloudVAE
from models.latent_critic import LatentTokenCritic, LatentRefiner
from models.latent_diffusion import LatentNormalizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K, LATENT, NTOK = 512, 128, 8
TD = LATENT // NTOK


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
    num_latent_tokens=NTOK, latent_dim=LATENT, disc_warmup_epochs=0, lambda_adv=1e-2,
    lambda_adv_ramp_epochs=0, n_critic=2, lambda_gp=10.0, eps_drift=1e-2, gamma_z=1.0)

vae = PointCloudVAE(latent_dim=LATENT, num_decoded_points=K, num_latent_tokens=NTOK,
                    internal_dim=64, num_heads=4, num_dec_blocks=2).to(device).eval()
for p in vae.parameters():
    p.requires_grad_(False)
refiner = LatentRefiner(TD, dim=64, num_heads=4, num_layers=2, num_tokens=NTOK, max_dz=0.5).to(device)
critic = LatentTokenCritic(TD, dim=64, num_heads=4, num_layers=2, num_tokens=NTOK).to(device)
norm = LatentNormalizer(LATENT).to(device)
og = optim.AdamW(refiner.parameters(), lr=3e-4)
od = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))
w = SummaryWriter("/tmp/smoke_latent_tb")
s = TL.train_epoch(vae, refiner, critic, norm, og, od, fake_loader(), 0, args, w, device)
w.close()
print(f"OK cd={s['cd']:.4g} dz={s['dz']:.3g} adv={s['adv']:.4g} cos={s['cos']}")
vcd = TL.validate(vae, refiner, norm, fake_loader(2), args, device)
print(f"val OK val_cd={vcd:.4g}")
print("LATENT SMOKE PASSED")
