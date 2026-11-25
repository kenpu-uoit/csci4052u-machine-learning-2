"""Building blocks for the Assignment 3 variational autoencoder.

Everything here is provided. You assemble these pieces in the notebook; you do
not edit this file.

The symbols follow Chapter 17 of the notes:

    encoder   g[x, theta]  ->  (mu, log sigma^2)      the variational posterior
    decoder   f[z, phi]    ->  x_hat                  the conditional mean of x

    Pr(x|z,phi) = Norm_x[ f[z,phi], sigma^2 I ]
    Pr(z)       = Norm_z[ 0, I ]
    q(z|x,theta)= Norm_z[ mu, Sigma ],  Sigma diagonal

`Encoder` returns the *logarithm* of the variance rather than the variance
itself: a linear layer may output a negative number and a variance may not, so
the network predicts log sigma^2 and the code exponentiates it.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

__all__ = [
    "D_Z", "SIGMA2",
    "Encoder", "Decoder",
    "reconstruction_loss", "kl_divergence",
    "load_z_samples",
]

# The latent dimension is fixed at 2 so that the whole latent space can be
# drawn. Part 4 depends on this.
D_Z = 2

# The decoder's spherical variance, sigma^2 in Pr(x|z,phi). A constant, not a
# learned parameter. It sets the exchange rate between the two loss terms.
SIGMA2 = 0.1


class Encoder(nn.Module):
    """g[x, theta]: an image to the mean and log-variance of q(z|x,theta).

    Two stride-2 convolutions halve the spatial size each time, 28 -> 14 -> 7,
    then two linear heads read the same trunk.

        x        (B, 1, 28, 28)
        ->       (B, 32, 14, 14)
        ->       (B, 64,  7,  7)
        ->       (B, 256)
        -> mu, logvar, each (B, d_z)
    """

    def __init__(self, d_z: int = D_Z):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, d_z)
        self.fc_logvar = nn.Linear(256, d_z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 1, 28, 28) -> (mu, logvar), each (B, d_z)."""
        h = self.trunk(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """f[z, phi]: a latent vector to the conditional mean of the image.

    The mirror image of `Encoder`: two transposed convolutions double the
    spatial size each time, 7 -> 14 -> 28. The closing sigmoid puts the output
    on the same [0, 1] scale as the input pixels.

        z        (B, d_z)
        ->       (B, 64,  7,  7)
        ->       (B, 32, 14, 14)
        -> x_hat (B,  1, 28, 28)
    """

    def __init__(self, d_z: int = D_Z):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(d_z, 256), nn.ReLU(),
            nn.Linear(256, 64 * 7 * 7), nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d_z) -> x_hat: (B, 1, 28, 28), pixels in [0, 1]."""
        h = self.project(z)
        h = h.view(-1, 64, 7, 7)
        return self.deconv(h)


def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor,
                        sigma2: float = SIGMA2) -> torch.Tensor:
    """-log Pr(x | z*, phi), averaged over the batch.

    With Pr(x|z,phi) = Norm_x[f[z,phi], sigma^2 I] the log density is
    -||x - x_hat||^2 / (2 sigma^2) plus a constant that depends on neither
    theta nor phi. Dropping that constant,

        -log Pr(x|z*,phi) = (1 / 2 sigma^2) * sum_d (x_d - x_hat_d)^2

    The sum runs over all 784 pixels of one image; the mean runs over the
    batch. Summing one and averaging the other would silently reweight this
    term against the KL by a factor of 784.

    Returns a scalar.
    """
    per_image = ((x - x_hat) ** 2).flatten(1).sum(1)
    return (per_image / (2 * sigma2)).mean()


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """D_KL[ q(z|x,theta) || Norm_z[0, I] ], averaged over the batch.

    Equation 17.24 of the notes, with Sigma diagonal. Writing
    Sigma = diag(sigma_1^2, ..., sigma_Dz^2), the trace is the sum of the
    sigma_j^2 and log det Sigma is the sum of the log sigma_j^2, so

        D_KL = 0.5 * sum_j ( sigma_j^2 + mu_j^2 - 1 - log sigma_j^2 )

    The divergence is zero exactly when mu = 0 and Sigma = I, that is when the
    approximate posterior has become the prior.

    Returns a scalar.
    """
    per_image = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1)
    return per_image.mean()


def load_z_samples(path: str | Path = "z_samples.pt") -> torch.Tensor:
    """The precomputed 10x10 sweep of the latent plane, as a (100, 2) tensor.

    The prior is Norm_z[0, I], so an even sweep of the latent space is even in
    *probability*, not in z: the ten coordinates on each axis are the midpoints
    of ten equal probability slices, mapped through the inverse normal CDF.
    They are +/-1.645, +/-1.036, +/-0.674, +/-0.385, +/-0.126.

    A linspace over [-3, 3] would instead spend its outer cells in the tails,
    where the prior places almost no mass and the model has seen almost no
    training examples, so those cells decode to smears.

    The rows are ordered for display as a 10x10 grid read like a graph: z1
    increases left to right along a row, z2 increases bottom to top up the
    rows. Row r, column c of the picture is entry 10*r + c.

    `make_z_samples.py` is the script that wrote the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. It ships beside the notebook; on Colab the setup "
            "cell downloads it. Re-run the setup cell."
        )
    z = torch.load(path, map_location="cpu", weights_only=True)
    if z.shape != (100, D_Z):
        raise ValueError(f"{path} holds shape {tuple(z.shape)}, expected (100, {D_Z})")
    return z
