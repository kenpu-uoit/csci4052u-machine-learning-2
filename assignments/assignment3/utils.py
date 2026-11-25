"""Rendering and checking helpers for Assignment 3.

Everything here is provided. You call these from the notebook; you do not edit
this file.

Rendering
    show_grid(images, ...)          a square grid of greyscale images
    show_scatter(mu, labels, ...)   the latent plane, coloured by digit
    to_frames(images)               float images -> uint8 frames for a GIF

Checking
    Every `check_*` prints `[ok]` and a short summary when it is satisfied, and
    raises `AssertionError` with a diagnostic when it is not. A failure names
    what was expected and what arrived, so read the message rather than the
    traceback.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from components import D_Z, SIGMA2, kl_divergence, reconstruction_loss

__all__ = [
    "show_grid", "show_scatter", "to_frames",
    "check_vae", "check_loss_fn", "check_history", "check_decoded_grid",
    "check_latents", "check_path", "check_gif",
]

GRID = 10


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _as_images(images) -> np.ndarray:
    a = images.detach().cpu().numpy() if torch.is_tensor(images) else np.asarray(images)
    return a.reshape(-1, 28, 28)


def show_grid(images, titles=None, nrow: int = GRID, figsize=(8, 8),
              suptitle: str | None = None) -> None:
    """Draw a square grid of greyscale images.

    images : (N, 28, 28) or (N, 1, 28, 28), N == nrow**2
    titles : optional list of N strings, one per panel
    """
    ims = _as_images(images)
    if len(ims) < nrow * nrow:
        raise ValueError(f"{len(ims)} images, need {nrow * nrow} for a {nrow}x{nrow} grid")
    fig, axes = plt.subplots(nrow, nrow, figsize=figsize)
    for k, ax in enumerate(axes.flat):
        ax.imshow(ims[k], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if titles is not None:
            ax.set_title(str(titles[k]), fontsize=8, pad=2)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    plt.show()


def show_scatter(mu, labels, extra=None, extra_label: str = "path",
                 title: str = "The latent plane", figsize=(7, 6),
                 annotate_centroids: bool = False) -> None:
    """Scatter encoded means on the latent plane, coloured by digit.

    mu     : (N, 2) encoded means
    labels : (N,) digit labels, 0-9
    extra  : optional (M, 2) of further points, drawn over the top in black
    """
    mu = mu.detach().cpu().numpy() if torch.is_tensor(mu) else np.asarray(mu)
    labels = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(mu[:, 0], mu[:, 1], c=labels, cmap="tab10", s=3, alpha=0.6)
    fig.colorbar(sc, ax=ax, ticks=range(10), label="digit")

    if annotate_centroids:
        for d in range(10):
            sel = labels == d
            if sel.any():
                cx, cy = mu[sel].mean(0)
                ax.text(cx, cy, str(d), fontsize=14, fontweight="bold",
                        ha="center", va="center",
                        bbox=dict(boxstyle="circle,pad=0.15", fc="white", alpha=0.75))

    if extra is not None:
        e = extra.detach().cpu().numpy() if torch.is_tensor(extra) else np.asarray(extra)
        ax.plot(e[:, 0], e[:, 1], "-", color="k", lw=1.0, alpha=0.6)
        ax.scatter(e[:, 0], e[:, 1], c="k", marker="+", s=45, label=extra_label)
        ax.legend(loc="upper left")

    ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def to_frames(images, scale: int = 4) -> np.ndarray:
    """Float images in [0, 1] -> uint8 frames ready for imageio.

    images : (N, 1, 28, 28) or (N, 28, 28)
    scale  : nearest-neighbour magnification; 28x28 is tiny in a GIF viewer

    Returns (N, 28*scale, 28*scale) of uint8.
    """
    ims = np.clip(_as_images(images), 0.0, 1.0)
    frames = (ims * 255).astype(np.uint8)
    if scale > 1:
        frames = frames.repeat(scale, axis=1).repeat(scale, axis=2)
    return frames


# --------------------------------------------------------------------------
# checking
# --------------------------------------------------------------------------

def _ok(msg: str) -> None:
    print(f"[ok] {msg}")


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def check_vae(vae) -> None:
    """The assembled model: named sub-modules, a differentiable sample, shapes."""
    for name in ("encoder", "decoder"):
        if not hasattr(vae, name):
            _fail(f"the model has no attribute `{name}`. train.encode_dataset and "
                  f"train.decode_z reach for `vae.encoder` and `vae.decoder`, so the "
                  f"sub-modules must carry exactly those names.")
    if not isinstance(vae.encoder, torch.nn.Module):
        _fail("vae.encoder is not a Module - assign an Encoder() instance")
    if not isinstance(vae.decoder, torch.nn.Module):
        _fail("vae.decoder is not a Module - assign a Decoder() instance")

    # follow the model onto whatever device it was moved to
    try:
        device = next(vae.parameters()).device
    except StopIteration:
        _fail("the model has no parameters - were the sub-modules assigned to "
              "`self.encoder` and `self.decoder`?")

    x = torch.rand(5, 1, 28, 28, device=device)
    out = vae(x)
    if not (isinstance(out, tuple) and len(out) == 3):
        _fail("forward must return the triple (x_hat, mu, logvar); got "
              f"{type(out).__name__}" + (f" of length {len(out)}"
                                         if isinstance(out, tuple) else ""))
    x_hat, mu, logvar = out
    if tuple(x_hat.shape) != (5, 1, 28, 28):
        _fail(f"x_hat has shape {tuple(x_hat.shape)}, expected {(5, 1, 28, 28)}")
    for nm, t in (("mu", mu), ("logvar", logvar)):
        if tuple(t.shape) != (5, D_Z):
            _fail(f"{nm} has shape {tuple(t.shape)}, expected {(5, D_Z)}")
    if torch.allclose(mu, logvar):
        _fail("mu and logvar are identical - return both outputs of the encoder, "
              "in that order")
    if float(x_hat.min()) < 0 or float(x_hat.max()) > 1:
        _fail("x_hat left [0, 1] - the Decoder's closing sigmoid was bypassed")

    # the reparameterization must be differentiable and must actually sample
    mu_t = torch.zeros(4, D_Z, device=device, requires_grad=True)
    lv_t = torch.zeros(4, D_Z, device=device, requires_grad=True)
    z1 = vae.reparameterize(mu_t, lv_t)
    if tuple(z1.shape) != (4, D_Z):
        _fail(f"reparameterize returned {tuple(z1.shape)}, expected {(4, D_Z)}")
    if z1.grad_fn is None:
        _fail("z* has no grad_fn: the sample is detached from mu and logvar, so the "
              "encoder would never receive a gradient. Build z* as "
              "mu + std * torch.randn_like(mu); torch.normal(mu, std) will not do.")
    g, = torch.autograd.grad(z1.sum(), mu_t, retain_graph=True)
    if not torch.allclose(g, torch.ones_like(g)):
        _fail("d(sum z*)/d(mu) should be all ones for z* = mu + sigma * eps")
    if torch.allclose(vae.reparameterize(mu_t, lv_t), vae.reparameterize(mu_t, lv_t)):
        _fail("two calls returned the same z* - you returned mu instead of sampling")

    n = sum(p.numel() for p in vae.parameters())
    _ok(f"encoder, decoder, a differentiable z* and a (x_hat, mu, logvar) "
        f"forward; {n:,} parameters")


def check_loss_fn(loss_fn) -> None:
    """The objective: three scalars, differentiable, agreeing with the components."""
    torch.manual_seed(0)
    x = torch.rand(8, 1, 28, 28)
    x_hat = torch.rand(8, 1, 28, 28, requires_grad=True)
    mu = torch.randn(8, D_Z, requires_grad=True)
    logvar = torch.randn(8, D_Z, requires_grad=True) * 0.1

    out = loss_fn(x, x_hat, mu, logvar)
    if not (isinstance(out, tuple) and len(out) == 3):
        _fail("loss_fn must return the triple (loss, recon, kl) - train_vae unpacks "
              "three values, and plot_metrics needs the two components separately")
    loss, recon, kl = out
    for nm, t in (("loss", loss), ("recon", recon), ("kl", kl)):
        if not torch.is_tensor(t) or t.ndim != 0:
            _fail(f"{nm} must be a scalar tensor, got {type(t).__name__}"
                  + (f" of shape {tuple(t.shape)}" if torch.is_tensor(t) else ""))
    if loss.grad_fn is None:
        _fail("loss has no grad_fn - nothing would train")

    want_r = reconstruction_loss(x, x_hat, SIGMA2)
    want_k = kl_divergence(mu, logvar)
    if not torch.allclose(recon, want_r, rtol=1e-4):
        _fail(f"recon is {float(recon):.2f}, but components.reconstruction_loss gives "
              f"{float(want_r):.2f}. Call the provided function rather than "
              "reimplementing it.")
    if not torch.allclose(kl, want_k, rtol=1e-4):
        _fail(f"kl is {float(kl):.3f}, but components.kl_divergence gives "
              f"{float(want_k):.3f}. Call the provided function.")
    if not torch.allclose(loss, want_r + want_k, rtol=1e-4):
        _fail(f"loss is {float(loss):.2f}, expected recon + kl = "
              f"{float(want_r + want_k):.2f}. The KL is ADDED: you minimise the "
              "negative ELBO, so the penalty is positive.")
    _ok(f"loss = recon + kl, all three scalar and differentiable "
        f"(recon {float(recon):.1f}, kl {float(kl):.2f} on random input)")


def check_history(history) -> None:
    """The training run: enough epochs, a falling loss, a latent still in use."""
    for col in ("epoch", "loss", "recon", "kl"):
        if col not in history.columns:
            _fail(f"history has no column {col!r}; it has {list(history.columns)}")
    if len(history) < 2:
        _fail(f"{len(history)} epoch recorded - training did not run")
    first, last = float(history["loss"].iloc[0]), float(history["loss"].iloc[-1])
    if not last < first:
        _fail(f"loss rose, {first:.1f} -> {last:.1f}. Check the sign of the KL term: "
              "it is added, because you minimise the negative ELBO.")
    if last > 300:
        _fail(f"final loss {last:.1f} is far too high; a correct run lands near 160.")
    kl_last = float(history["kl"].iloc[-1])
    if kl_last < 0.5:
        _fail(f"the KL collapsed to {kl_last:.3f}: the encoder is returning the prior "
              "and the latent carries no information (posterior collapse)")
    _ok(f"{len(history)} epochs; loss {first:.1f} -> {last:.1f}, "
        f"recon {float(history['recon'].iloc[0]):.1f} -> "
        f"{float(history['recon'].iloc[-1]):.1f}, "
        f"kl {float(history['kl'].iloc[0]):.2f} -> {kl_last:.2f}")


def check_decoded_grid(z, images) -> None:
    """The decoded sweep: the shipped latents, and output that varies with z."""
    from components import load_z_samples
    want = load_z_samples()
    if tuple(z.shape) != (GRID * GRID, D_Z):
        _fail(f"z has shape {tuple(z.shape)}, expected {(GRID * GRID, D_Z)}")
    if not torch.allclose(z.cpu(), want, atol=1e-5):
        _fail("these are not the latents in z_samples.pt. Load them with "
              "components.load_z_samples() rather than building your own grid.")
    if tuple(images.shape) != (GRID * GRID, 1, 28, 28):
        _fail(f"decoded images have shape {tuple(images.shape)}, expected "
              f"{(GRID * GRID, 1, 28, 28)}")
    if float(images.min()) < 0 or float(images.max()) > 1:
        _fail("decoded pixels left [0, 1]")
    spread = float(images.flatten(1).mean(1).std())
    if spread < 0.01:
        _fail(f"all 100 outputs have nearly the same brightness (std {spread:.4f}) - "
              "the decoder is ignoring z")
    _ok(f"100 images decoded from z_samples.pt; ink varies by {spread:.3f} across "
        "the grid")


def check_latents(mu, labels, n_expected: int | None = None) -> None:
    """The encoded dataset: one 2-D point per image, and classes that separate."""
    if tuple(mu.shape)[1:] != (D_Z,):
        _fail(f"mu has shape {tuple(mu.shape)}, expected (N, {D_Z}) - keep the mean "
              "only, not the pair (mu, logvar)")
    if len(labels) != len(mu):
        _fail(f"{len(mu)} means but {len(labels)} labels")
    if n_expected is not None and len(mu) != n_expected:
        _fail(f"{len(mu)} points, expected {n_expected} - encode the whole dataset")
    present = sorted({int(v) for v in labels})
    if present != list(range(10)):
        _fail(f"labels cover {present}, expected all ten digits")
    centres = torch.stack([mu[labels == d].mean(0) for d in range(10)])
    spread = float(torch.cdist(centres, centres).max())
    if spread < 1.0:
        _fail(f"the ten class centres span only {spread:.2f} - the latent is not "
              "separating the digits. Was the trained model used?")
    _ok(f"{len(mu):,} encoded means, all ten classes present, "
        f"centres span {spread:.2f}")


def check_path(path, centroids=None, n_expected: int = 50) -> None:
    """The interpolation path: the right length, and it visits the centroids."""
    if tuple(path.shape) != (n_expected, D_Z):
        _fail(f"path has shape {tuple(path.shape)}, expected {(n_expected, D_Z)}")
    steps = torch.linalg.norm(path[1:] - path[:-1], dim=1)
    if float(steps.min()) <= 0:
        _fail("the path repeats a point - it should advance at every step")
    if centroids is not None:
        # every centroid should have some path point near it
        d = torch.cdist(centroids, path).min(dim=1).values
        far = [int(i) for i, v in enumerate(d) if float(v) > 0.6]
        if far:
            _fail(f"the path never passes close to the centroids for {far}. It should "
                  "run 0 -> 1 -> ... -> 9 through all ten in order.")
    total = float(steps.sum())
    _ok(f"{len(path)} points along a path of length {total:.2f}, "
        f"steps {float(steps.min()):.3f}-{float(steps.max()):.3f}")


def check_gif(filename, n_expected: int = 50) -> None:
    """The animation: a real GIF on disk with the right number of frames."""
    p = Path(filename)
    if not p.exists():
        _fail(f"{p} was not written")
    if p.stat().st_size < 1024:
        _fail(f"{p} is only {p.stat().st_size} bytes - it is unlikely to hold "
              f"{n_expected} frames")
    with open(p, "rb") as fh:
        magic = fh.read(6)
    if magic not in (b"GIF87a", b"GIF89a"):
        _fail(f"{p} does not start with a GIF header (got {magic!r}) - check the "
              "file extension passed to imageio")
    try:
        import imageio.v3 as iio
        frames = iio.imread(p, index=None)
        n = len(frames)
    except Exception as exc:                      # pragma: no cover
        _ok(f"{p} written, {p.stat().st_size / 1024:.0f} KB (frames not counted: {exc})")
        return
    if n != n_expected:
        _fail(f"{p} holds {n} frames, expected {n_expected}")
    _ok(f"{p} written: {n} frames, {frames.shape[1]}x{frames.shape[2]}, "
        f"{p.stat().st_size / 1024:.0f} KB")
