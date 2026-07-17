"""In-process seeding wrapper so the latent-diffusion run is reproducible
without editing the tracked training script. Usage:
    python seed_then_train.py <all the train_diffusion_latent.py flags>
Seeds torch/numpy/random to 42, then runs training/train_diffusion_latent.py as __main__.
"""
import sys
import random
import numpy as np
import torch
import runpy

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# hand the remaining argv to the training script's argparse
sys.argv = ["training/train_diffusion_latent.py"] + sys.argv[1:]
runpy.run_path("training/train_diffusion_latent.py", run_name="__main__")
