#!/usr/bin/env python
"""Write z_samples.pt: the 10x10 even sweep of the latent plane.

    uv run python ../assignments/assignment3/make_z_samples.py

Run from `labs/`, so that the project's torch is on the path. The file is
committed, so this script exists to document how it was made and to let the
grid be regenerated if D_Z or GRID ever change.
"""

from pathlib import Path

import torch

HERE = Path(__file__).parent
GRID = 10
D_Z = 2

# The prior is Norm_z[0, I], so "even" means even in probability. Take the
# midpoints of GRID equal probability slices and map them through the inverse
# normal CDF.
probs = torch.linspace(0.5 / GRID, 1 - 0.5 / GRID, GRID)
coords = torch.distributions.Normal(0.0, 1.0).icdf(probs)

# Row-major, ordered for display: z1 increases left to right along a row, z2
# increases bottom to top up the rows, so the top row carries the largest z2.
z_samples = torch.stack([
    torch.tensor([float(z1), float(z2)])
    for z2 in coords.flip(0)
    for z1 in coords
])

assert z_samples.shape == (GRID * GRID, D_Z)
torch.save(z_samples, HERE / "z_samples.pt")

print(f"coordinates: {[round(float(v), 3) for v in coords]}")
print(f"wrote {HERE / 'z_samples.pt'}: {tuple(z_samples.shape)}")
print(f"first row z2={float(z_samples[0, 1]):.3f}, last row z2={float(z_samples[-1, 1]):.3f}")
