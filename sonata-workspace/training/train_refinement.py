"""
Training Script for Refinement Network

Trains the refinement network to densify coarse diffusion outputs.
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.refinement_dataset import RefinementDataset, collate_refinement
from models.refinement_net import RefinementNetwork, chamfer_distance
from utils.checkpoint import save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train refinement network")
    parser.add_argument("--data_path", type=str, default=os.path.expanduser("~/Simon_ws/dataset/SemanticKITTI/dataset"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--coarse_voxel_size", type=float, default=0.1)
    parser.add_argument("--fine_voxel_size", type=float, default=0.05)
    parser.add_argument("--up_factor", type=int, default=6)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="checkpoints/refinement")
    parser.add_argument("--log_dir", type=str, default="logs/refinement")
    parser.add_argument("--save_freq", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def train_epoch(model, loader, optimizer, epoch, args, writer):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        coarse = batch["coarse"].cuda()
        dense = batch["dense"].cuda()
        coarse_mask = batch["coarse_mask"].cuda()
        coarse_lengths = batch["coarse_lengths"]
        dense_lengths = batch["dense_lengths"]

        batch_size = coarse.shape[0]
        losses = []

        for b in range(batch_size):
            n_c = coarse_lengths[b].item()
            n_d = dense_lengths[b].item()
            if n_c < 10 or n_d < 10:
                continue

            c = coarse[b, :n_c]  # (N, 3)
            d = dense[b, :n_d]  # (M, 3)

            refined = model(c)  # (N * up_factor, 3)
            loss = chamfer_distance(refined, d)
            losses.append(loss)

        if len(losses) == 0:
            continue

        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if n_batches > 0:
        avg_loss = total_loss / n_batches
        writer.add_scalar("train/loss", avg_loss, epoch)
        return avg_loss
    return 0.0


def main():
    args = parse_args()
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        # Flatten nested config
        flat = cfg.get("refinement", cfg) if isinstance(cfg.get("refinement"), dict) else cfg
        for k, v in flat.items():
            if hasattr(args, k):
                setattr(args, k, v)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    print("Loading refinement dataset...")
    train_dataset = RefinementDataset(
        root=args.data_path,
        split="train",
        coarse_voxel_size=args.coarse_voxel_size,
        fine_voxel_size=args.fine_voxel_size,
        use_ground_truth_maps=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_refinement,
        pin_memory=True,
    )

    print("Building refinement model...")
    model = RefinementNetwork(up_factor=args.up_factor).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0

    if args.resume and os.path.exists(args.resume):
        ckpt = load_checkpoint(args.resume)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        start_epoch = ckpt.get("epoch", 0) + 1
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Resumed from epoch {start_epoch}")

    best_loss = float("inf")
    for epoch in range(start_epoch, args.num_epochs):
        avg_loss = train_epoch(model, train_loader, optimizer, epoch, args, writer)
        print(f"Epoch {epoch} train loss: {avg_loss:.6f}")

        if (epoch + 1) % args.save_freq == 0:
            path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pth")
            save_checkpoint(path, model, optimizer=optimizer, epoch=epoch)

        if avg_loss < best_loss and avg_loss > 0:
            best_loss = avg_loss
            save_checkpoint(
                os.path.join(args.output_dir, "best_model.pth"),
                model, optimizer=optimizer, epoch=epoch
            )

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
