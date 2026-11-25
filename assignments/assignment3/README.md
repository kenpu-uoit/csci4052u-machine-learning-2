# A convolutional VAE toolkit

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kenpu-uoit/csci4052u-machine-learning-2/blob/main/assignments/assignment3/assignment3.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kenpu-uoit/csci4052u-machine-learning-2/blob/main/assignments/assignment3/assignment3_bonus.ipynb) — bonus notebook

Provided building blocks for a variational autoencoder over 28×28 greyscale images
with a two-dimensional latent space. The modules supply the layers, the training
loop, the plotting and the checks; you supply the model and the loss.

| file | holds |
|---|---|
| `components.py` | the encoder, the decoder, the two loss terms, the latent grid loader |
| `train.py` | the training loop, metric plotting, inference, save/load |
| `utils.py` | rendering helpers and every `check_*` |
| `z_samples.pt` | 100 precomputed latent vectors, a 10×10 sweep of the plane |
| `make_z_samples.py` | the script that wrote `z_samples.pt` |

## Completing the notebook

1. **Open it in Colab** with the badge above, then **File → Save a copy in Drive**
   before doing any work. Colab discards an unsaved session.
2. **Turn on a GPU** — *Runtime → Change runtime type → T4 GPU*. Training takes a
   minute or two on a GPU and roughly twenty times that without one.
3. **Run the setup cells.** They install what Colab lacks, download the support
   files beside the notebook, and fetch the dataset.
4. **Fill in the four cells marked `# YOUR CODE HERE`:**
   - assemble `Encoder` and `Decoder` into a model, joined by the reparameterization
   - write the loss from the two provided terms, returning `(loss, recon, kl)`
   - call the training function, then plot the metrics it returns
   - load the latent grid, decode it, and display it
5. **Run the CHECK cell under each one.** Every check must print `[ok]`. A failure
   names what was expected and what arrived — read the message, not the traceback.
6. **Answer the markdown cells headed “Your answer.”** Two or three sentences each.
7. **Run the whole notebook top to bottom and leave every output in place.** The
   figures are the evidence; a notebook submitted with cleared outputs is read as
   code alone.
8. The last cell writes `vae.pt`. Keep it — the bonus notebook reads it.
9. **Submit the `.ipynb`** with its outputs intact.

## Completing the bonus notebook

Run the main notebook to completion first: this one loads `vae.pt` and trains
nothing. Keep both notebooks in the same directory.

1. **Open, save a copy and select a GPU** as above, then run the setup cells.
2. **Fill in the four cells marked `# YOUR CODE HERE`:**
   - encode the dataset and scatter the latent points by label
   - compute the per-class centroids, and lay an evenly spaced path through them
   - decode every point on the path
   - write the sequence out as an animated GIF and display it inline
3. **Run the CHECK cell under each one**, and answer the “Your answer” cell.
4. **Submit the `.ipynb` with its outputs intact, and the `.gif` beside it.**

---

## The helpers

```python
from components import D_Z, SIGMA2, Encoder, Decoder
from components import reconstruction_loss, kl_divergence, load_z_samples
from train import train_vae, plot_metrics, encode_dataset, decode_z
from train import save_trained, load_trained
from utils import show_grid, show_scatter, to_frames
```

Two constants: `D_Z = 2`, the latent dimension, and `SIGMA2 = 0.1`, the decoder's
spherical variance.

---

## components.py

**`Encoder(d_z=D_Z)`** — `(B, 1, 28, 28)` → `(mu, logvar)`, each `(B, d_z)`.
Two stride-2 convolutions, 28 → 14 → 7, then two linear heads on a shared trunk.
It returns the **log** of the variance: a linear layer may emit a negative number
and a variance may not.

**`Decoder(d_z=D_Z)`** — `(B, d_z)` → `(B, 1, 28, 28)`. The mirror image: two
transposed convolutions, 7 → 14 → 28, closing on a sigmoid so the output lands on
the same `[0, 1]` scale as the input.

**`reconstruction_loss(x, x_hat, sigma2=SIGMA2)`** → scalar. Squared error summed
over the 784 pixels, divided by $2\sigma^2$, then averaged over the batch.

$$
\frac{1}{2\sigma^2}\sum_{d} \left( x_d - \hat{x}_d \right)^2
$$

**`kl_divergence(mu, logvar)`** → scalar. Divergence from the diagonal Gaussian to a
standard normal, summed over latent dimensions, averaged over the batch. Zero
exactly when `mu = 0` and the variance is 1.

$$
\frac{1}{2}\sum_{j} \left( \sigma_j^2 + \mu_j^2 - 1 - \log \sigma_j^2 \right)
$$

Both sum over their own dimensions and average over the batch. Averaging one where
the formula says sum reweights it against the other by a factor of 784.

**`load_z_samples(path="z_samples.pt")`** → `(100, 2)`. An even sweep of the latent
plane — even in *probability*, not in `z`: the ten coordinates per axis are the
midpoints of ten equal probability slices under a standard normal, ±1.645, ±1.036,
±0.674, ±0.385, ±0.126. Rows are ordered for a 10×10 grid read like a graph, `z1`
rightward and `z2` upward.

---

## train.py

**`train_vae(vae, loss_fn, dataset, *, epochs=12, batch_size=256, lr=1e-3, num_workers=2, seed=0)`**
→ `DataFrame` with one row per epoch and columns `epoch`, `loss`, `recon`, `kl`.

Wraps your module and loss in a LightningModule and runs Adam. The two arguments
it requires of you:

- `vae.forward(x)` returns the triple `(x_hat, mu, logvar)`
- `loss_fn(x, x_hat, mu, logvar)` returns the triple `(loss, recon, kl)`

Metrics are written to `logs/metrics.csv`, **overwritten on every run** — there is
no version-numbered directory to sift through.

**`plot_metrics(history)`** — the two loss terms against epoch, in two panels.
Separate panels because reconstruction sits near 150 and the KL near 7; on a shared
axis the KL is a flat line on the floor.

**`encode_dataset(vae, dataset, *, batch_size=1000, num_workers=2)`** → `(mu, labels)`,
one 2-D point per image. Keeps the mean only.

**`decode_z(vae, z)`** → `(N, 1, 28, 28)` on the CPU. Handles `eval()`, `no_grad()`
and the device.

**`save_trained(vae, path="vae.pt")`** / **`load_trained(path="vae.pt", device=None)`**
— store and reopen a trained model. Only the two sub-modules are written, never the
assembled class, so a checkpoint can be reopened without knowing how the model was
put together. `load_trained` returns an object carrying `.encoder` and `.decoder`,
which is all `encode_dataset` and `decode_z` need.

> `encode_dataset`, `decode_z` and `load_trained` reach for **`vae.encoder`** and
> **`vae.decoder`**. Give your sub-modules exactly those names.

---

## utils.py

**Rendering**

| | |
|---|---|
| `show_grid(images, titles=None, nrow=10, figsize=(8, 8), suptitle=None)` | a square grid of greyscale images |
| `show_scatter(mu, labels, extra=None, extra_label="path", title=..., annotate_centroids=False)` | the latent plane coloured by label; `extra` draws a further set of points over the top |
| `to_frames(images, scale=4)` | float images → `uint8` frames for `imageio`, magnified |

**Checks** — each prints `[ok]` with a summary, or raises `AssertionError` naming
what was expected and what arrived.

| | checks |
|---|---|
| `check_vae(vae)` | named sub-modules, shapes, and that the latent sample is differentiable |
| `check_loss_fn(loss_fn)` | three scalars, gradients, values agreeing with the provided terms |
| `check_history(history)` | enough epochs, a falling loss, a latent still carrying information |
| `check_decoded_grid(z, images)` | the shipped latents were used, and the output varies with `z` |
| `check_latents(mu, labels, n_expected=None)` | one point per image, and classes that separate |
| `check_path(path, centroids=None, n_expected=50)` | the right length, advancing, visiting the centroids |
| `check_gif(filename, n_expected=50)` | a real GIF on disk with the right frame count |

---

## Putting it together

```python
class VAE(nn.Module):
    def __init__(self, d_z=D_Z):
        super().__init__()
        self.encoder = Encoder(d_z)          # the names train.py expects
        self.decoder = Decoder(d_z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        return self.decoder(self.reparameterize(mu, logvar)), mu, logvar


def vae_loss(x, x_hat, mu, logvar):
    recon = reconstruction_loss(x, x_hat, SIGMA2)
    kl = kl_divergence(mu, logvar)
    return recon + kl, recon, kl


vae = VAE().to(DEVICE)
history = train_vae(vae, vae_loss, train_set)
plot_metrics(history)

show_grid(decode_z(vae, load_z_samples()))
save_trained(vae, "vae.pt")
```

---

## Two conventions

1. **The encoder predicts `logvar`, not the variance.** Exponentiate to recover it;
   `std = exp(0.5 * logvar)`.
2. **The loss is the negative ELBO, so it is minimised and the KL is added.** Get
   the sign wrong and the loss rises; `check_history` says so in those words.

The latent sample must be built as `mu + std * torch.randn_like(mu)`.
`torch.normal(mu, std)` draws from the same distribution and carries no gradient
back to `mu`, so the encoder would never learn. `check_vae` differentiates the
sample and rejects a detached one.
