#!/usr/bin/env python3
"""Fast integration smoke test for the critic redesign — no dataset needed.

Exercises train_epoch through every new code path (token critic, adaptive
adversarial weight, R1 penalty, conditional conditioning) plus the legacy
path, on synthetic in-memory batches. Catches wiring bugs before a real run.
"""
import os, sys, types
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

import training.train_vae_gan as T
from models.point_cloud_vae import PointCloudVAE
from models.discriminator import MultiScalePointDiscriminator
from models.critic_token import TokenCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 512


def fake_loader(n_batches=3, bsz=2):
    out = []
    for _ in range(n_batches):
        cc, cb, pc, pb = [], [], [], []
        for b in range(bsz):
            nc = torch.randint(600, 1200, (1,)).item()
            npp = torch.randint(300, 800, (1,)).item()
            cc.append(torch.randn(nc, 3))
            cb.append(torch.full((nc,), b, dtype=torch.long))
            pc.append(torch.randn(npp, 3))
            pb.append(torch.full((npp,), b, dtype=torch.long))
        out.append({
            "complete_coord": torch.cat(cc), "complete_batch": torch.cat(cb),
            "partial_coord": torch.cat(pc), "partial_batch": torch.cat(pb),
        })
    return out


def make_args(**over):
    a = types.SimpleNamespace(
        num_decoded_points=K, beta_kl=1e-3, center_gt=True, scale_gt=True,
        n_critic=2, lambda_gp=10.0, lambda_adv=1.0, lambda_adv_ramp_epochs=0,
        lambda_fm=0.0, disc_warmup_epochs=0, gradient_clip=1.0,
        ema_start_epoch=0, critic="token", critic_tokens=8, critic_dim=64,
        critic_layers=1, conditional_critic=False, adaptive_adv=True,
        adaptive_max=1e4, gp_mode="r1", log_grad_align=True,
    )
    for k, v in over.items():
        setattr(a, k, v)
    return a


def build_vae():
    return PointCloudVAE(latent_dim=128, num_decoded_points=K,
                         num_latent_tokens=8, internal_dim=64,
                         num_heads=4, num_dec_blocks=2).to(device)


def run(name, args, disc):
    vae = build_vae()
    og = optim.AdamW(vae.parameters(), lr=1e-4)
    od = optim.Adam(disc.parameters(), lr=1e-4, betas=(0.0, 0.9))
    w = SummaryWriter("/tmp/smoke_tb")
    s = T.train_epoch(vae, disc, og, od, fake_loader(), 0, args, w, device, ema=None)
    w.close()
    print(f"[{name}] OK  gen={s['gen']:.4g} cd={s['chamfer']:.4g} "
          f"adv={s['adv']:.4g} adapt_w={s.get('adapt_w'):.4g} cos={s.get('cos')}")


print("device:", device)
run("token+adaptive+r1+conditional",
    make_args(conditional_critic=True),
    TokenCritic(in_dim=3, dim=64, num_tokens=8, num_heads=4, num_layers=1, conditional=True).to(device))
run("token+adaptive+r1",
    make_args(),
    TokenCritic(in_dim=3, dim=64, num_tokens=8, num_heads=4, num_layers=1).to(device))
run("legacy multiscale+interp+fm (regression)",
    make_args(critic="multiscale", adaptive_adv=False, gp_mode="interp",
              lambda_adv=0.01, lambda_fm=1.0, log_grad_align=False),
    MultiScalePointDiscriminator(in_dim=3).to(device))
print("ALL SMOKE TESTS PASSED")
