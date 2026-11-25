"""Training and inference for the Assignment 3 variational autoencoder.

Everything here is provided. You call these functions from the notebook; you do
not edit this file.

    train_vae(vae, loss_fn, dataset)   fit the model, return the metric history
    plot_metrics(history)              the two loss terms against epoch
    encode_dataset(vae, dataset)       every image to its mu, plus the labels
    decode_z(vae, z)                   latent vectors to images

`train_vae` wraps whatever module you hand it in a LightningModule, so the
Lightning machinery stays out of the notebook. Your `vae` supplies `forward`
and your `loss_fn` supplies the objective; this file supplies the loop.

Metrics are written to `logs/metrics.csv` and that single file is overwritten
on every run. There is no run-versioned directory to sift through: the numbers
on disk are always the numbers from the last training.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import pandas as pd
import torch
from lightning.pytorch.loggers import CSVLogger

__all__ = ["LOG_DIR", "train_vae", "plot_metrics", "encode_dataset", "decode_z",
           "save_trained", "load_trained"]

LOG_DIR = Path("logs")

EPOCHS = 12
BATCH = 256
LR = 1e-3


class _LitVAE(L.LightningModule):
    """Internal. Drives the student's `vae` and `loss_fn` through Lightning."""

    def __init__(self, vae: torch.nn.Module, loss_fn, lr: float = LR):
        super().__init__()
        self.vae = vae
        self.loss_fn = loss_fn
        self.lr = lr

    def forward(self, x):
        return self.vae(x)

    def training_step(self, batch, batch_idx):
        x, _labels = batch          # a VAE is unsupervised; the labels go unused
        x_hat, mu, logvar = self.vae(x)
        loss, recon, kl = self.loss_fn(x, x_hat, mu, logvar)
        # The two components are logged separately, which is what makes the
        # left and right panels of plot_metrics possible.
        self.log_dict({"loss": loss, "recon": recon, "kl": kl},
                      on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.vae.parameters(), lr=self.lr)


def train_vae(vae: torch.nn.Module, loss_fn, dataset, *,
              epochs: int = EPOCHS, batch_size: int = BATCH,
              lr: float = LR, num_workers: int = 2,
              seed: int = 0) -> pd.DataFrame:
    """Fit `vae` on `dataset` and return one row of metrics per epoch.

    vae      : a module whose forward(x) returns (x_hat, mu, logvar)
    loss_fn  : loss_fn(x, x_hat, mu, logvar) -> (loss, recon, kl)
    dataset  : any torch Dataset yielding (image, label) pairs

    The returned DataFrame has columns `epoch`, `loss`, `recon` and `kl`, and
    is also on disk at `logs/metrics.csv`.
    """
    L.seed_everything(seed, workers=True)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        drop_last=True,
    )

    # One copy of the logs, never a version-numbered pile of them.
    if LOG_DIR.exists():
        shutil.rmtree(LOG_DIR)
    logger = CSVLogger(save_dir=".", name=str(LOG_DIR), version="")

    trainer = L.Trainer(
        max_epochs=epochs, accelerator="auto", devices=1,
        logger=logger, enable_checkpointing=False,
        enable_model_summary=False, log_every_n_steps=50,
    )
    trainer.fit(_LitVAE(vae, loss_fn, lr), loader)

    history = _read_metrics()
    last = history.iloc[-1]
    print(f"\nfinished {len(history)} epochs | "
          f"loss {last.loss:.1f} | recon {last.recon:.1f} | kl {last.kl:.2f}")
    print(f"metrics written to {LOG_DIR / 'metrics.csv'}")
    return history


def _read_metrics() -> pd.DataFrame:
    """Collapse the CSV logger's per-step rows into one row per epoch."""
    csv = LOG_DIR / "metrics.csv"
    if not csv.exists():
        raise FileNotFoundError(f"{csv} was not written; did training run?")
    df = pd.read_csv(csv)
    cols = [c for c in ("loss", "recon", "kl") if c in df.columns]
    out = (df.groupby("epoch")[cols].mean().reset_index())
    out["epoch"] = out["epoch"] + 1          # 1-based, for the plot
    return out


def plot_metrics(history: pd.DataFrame) -> None:
    """The two loss components against epoch, in two panels.

    Two panels rather than one: the reconstruction term sits near 150 and the
    KL near 7, so on a shared axis the KL is a flat line along the floor.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(history["epoch"], history["recon"], marker="o")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel(r"$-\log Pr(x \mid z^*, \phi)$")
    ax1.set_title("Reconstruction term")
    ax1.grid(alpha=0.3)

    ax2.plot(history["epoch"], history["kl"], marker="o", color="tab:red")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel(r"$D_{KL}[\, q \,\|\, Pr(z) \,]$")
    ax2.set_title("KL to the prior")
    ax2.grid(alpha=0.3)

    fig.suptitle("The two terms of the negative ELBO", fontsize=13)
    fig.tight_layout()
    plt.show()


@torch.no_grad()
def encode_dataset(vae: torch.nn.Module, dataset, *, batch_size: int = 1000,
                   num_workers: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode every image in `dataset`; return (mu, labels).

    Only the mean of q(z|x,theta) is kept — one point per image. mu is
    (N, d_z) and labels is (N,), both on the CPU.
    """
    device = _device_of(vae)
    vae.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         num_workers=num_workers)
    mus, labs = [], []
    for x, y in loader:
        mu, _logvar = vae.encoder(x.to(device))
        mus.append(mu.cpu())
        labs.append(y)
    return torch.cat(mus), torch.cat(labs)


@torch.no_grad()
def decode_z(vae: torch.nn.Module, z: torch.Tensor) -> torch.Tensor:
    """Decode latent vectors to images: (N, d_z) -> (N, 1, 28, 28) on the CPU."""
    device = _device_of(vae)
    vae.eval()
    return vae.decoder(z.to(device)).cpu()


def _device_of(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


# --------------------------------------------------------------------------
# saving and reloading a trained model
# --------------------------------------------------------------------------
#
# Only the two sub-modules are stored, never the assembled VAE class: Encoder
# and Decoder both live in components.py, so a checkpoint written by one
# notebook can be reopened by another without that notebook knowing how the
# VAE was put together.

class _Trained:
    """A loaded model: just enough of a VAE for encode_dataset and decode_z."""

    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.decoder.parameters()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        return self


def save_trained(vae, path: str | Path = "vae.pt") -> None:
    """Write the encoder and decoder weights to `path`."""
    torch.save({"encoder": vae.encoder.state_dict(),
                "decoder": vae.decoder.state_dict()}, path)
    print(f"saved the trained encoder and decoder to {path}")


def load_trained(path: str | Path = "vae.pt", device: str | None = None):
    """Rebuild a trained model from `path`.

    Returns an object carrying `.encoder` and `.decoder`, which is all that
    encode_dataset and decode_z require.
    """
    from components import Decoder, Encoder

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. It is written by the final cell of the basic "
            "assignment notebook; run that to completion first, and keep this "
            "notebook in the same directory."
        )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(path, map_location=device, weights_only=True)
    encoder, decoder = Encoder().to(device), Decoder().to(device)
    encoder.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])
    model = _Trained(encoder, decoder).eval()
    print(f"loaded the trained encoder and decoder from {path} onto {device}")
    return model
