# Assignment 3 — A Variational Autoencoder for MNIST

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kenpu-uoit/csci4052u-machine-learning-2/blob/main/assignments/assignment3/assignment3.ipynb) — the assignment

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kenpu-uoit/csci4052u-machine-learning-2/blob/main/assignments/assignment3/assignment3_bonus.ipynb) — the bonus notebook

**CSCI 4052U — Machine Learning II**
**Unit.** `chapter_17_variational_autoencoders` · **Builds on.** Chapter 12 (convolutions), Chapter 17 (the ELBO)
**Weight.** 100 marks, plus 20 bonus · **Expected effort.** 1–2 hours

---

## The question this assignment asks

Chapter 17 derives the evidence lower bound and assembles it into the variational
autoencoder. All of it is on paper. This assignment builds the thing and runs it on
MNIST.

**The parts are provided.** The convolutional encoder, the convolutional decoder, the
two terms of the loss and the training loop are all written for you. Your work is to
**assemble** them: wire the encoder and decoder into a VAE across the
reparameterization, combine the two loss terms into the objective, train, and decode.
That is the shape of the chapter, and it is what the assignment tests — not the
ability to retype a convolution stack.

The latent variable is **two-dimensional**, and that choice is the point rather than a
simplification for speed. A 2-D latent space can be drawn. Sweeping a grid across the
whole plane and decoding every cell produces one picture holding everything the model
learned to draw, and the gaps where it learned nothing.

---

## The files

| file | what it is |
|---|---|
| `assignment3.ipynb` | **the basic assignment**, 100 marks |
| `assignment3_bonus.ipynb` | **the bonus assignment**, 20 marks |
| `components.py` | the encoder, the decoder, the two loss terms, the latent loader |
| `train.py` | the Lightning training loop, plotting, encode/decode, save/load |
| `utils.py` | rendering helpers and every `check_*` function |
| `z_samples.pt` | the 100 precomputed latent vectors of Part 4 |
| `make_z_samples.py` | the script that wrote `z_samples.pt` |

**You edit only the notebooks.** The three `.py` files are provided and are not marked;
changing them to make a check pass is academic misconduct, and the marker runs against
the originals.

On Colab the first cell downloads the four support files automatically. Locally they
are already beside the notebook and the cell does nothing.

### What is provided, and what it corresponds to

| provided | is | from the notes |
|---|---|---|
| `Encoder()` | $g[x,\theta] \mapsto (\mu, \log\sigma^2)$ | 17.20 |
| `Decoder()` | $f[z,\phi] \mapsto \hat{x}$ | 17.1 |
| `reconstruction_loss(x, x_hat)` | $-\log Pr(x \mid z^{*},\phi)$ | 17.23, first term |
| `kl_divergence(mu, logvar)` | $D_{KL}[q(z \mid x,\theta)\,\Vert\,Pr(z)]$ | 17.24 |
| `train_vae(vae, loss_fn, dataset)` | the training loop, logging both terms | — |
| `plot_metrics(history)` | the two terms against epoch | — |
| `encode_dataset` / `decode_z` | inference | — |
| `load_z_samples()` | the latent sweep | — |

The one piece of the chapter that is **not** provided is the reparameterization,
equation 17.25. That is Part 1's job, and it is the step the chapter exists to teach.

---

## Before you start

**Turn on a GPU.** On Colab: *Runtime → Change runtime type → T4 GPU*. The assignment
trains one small model and uses well under 2 GB of GPU memory.

**Save a copy first.** *File → Save a copy in Drive*. Colab discards everything when
the session ends.

**Expected timings on a Colab T4**, measured:

| step | time |
|---|---|
| installing `lightning` | ~30 s, once |
| downloading MNIST (~11 MB) | ~10 s, once |
| Parts 1, 2 and 4, all checks | a few seconds |
| Part 3, training 12 epochs | 1–2 min |
| the whole bonus notebook | under a minute |

If training is taking twenty minutes, the setup cell printed `cpu` rather than `cuda`.

---

## The model, in one place

$$Pr(x|z,\phi) = \text{Norm}_x\!\left[f[z,\phi],\, \sigma^2 I\right],
\qquad Pr(z) = \text{Norm}_z[0, I],
\qquad q(z|x,\theta) = \text{Norm}_z[\mu, \Sigma]$$

The objective is the single-sample ELBO of **17.23**, negated so it can be descended:

$$-\text{ELBO} = \underbrace{\frac{1}{2\sigma^{2}}\sum_{d=1}^{784}(x_d - \hat{x}_d)^2}_{\texttt{reconstruction\_loss}}
\;+\; \underbrace{\frac{1}{2}\sum_{j=1}^{2}\left(\sigma_j^2 + \mu_j^2 - 1 - \log\sigma_j^2\right)}_{\texttt{kl\_divergence}}$$

**Two conventions the code uses throughout.**

1. **The encoder predicts $\log\sigma^2$, not $\Sigma$.** A linear layer may output a
   negative number and a variance may not, so the network predicts the logarithm and
   the code exponentiates it. $\Sigma$ is diagonal, so two numbers describe it.
2. **$\sigma^2$ is a fixed constant**, `SIGMA2 = 0.1` in `components.py`, not a learned
   parameter. It sets the exchange rate between the two terms: halving it doubles the
   weight on reconstruction. It is the one number worth experimenting with afterwards,
   and the last section says what to expect.

---

## The basic assignment, part by part

### Part 1 — Construct the VAE (30 marks)

A `VAE` module holding one `Encoder`, one `Decoder`, and the reparameterization
between them.

**Name the sub-modules exactly `self.encoder` and `self.decoder`.**
`train.encode_dataset` and `train.decode_z` reach for those attributes, as does the
check, and the bonus notebook's `load_trained` writes into them.

**`reparameterize`** is equation 17.25, $z^{*} = \mu + \sigma \odot \epsilon^{*}$ with
$\sigma = \exp[\tfrac12\log\sigma^2]$ and $\epsilon^{*}$ from `torch.randn_like(mu)`.

> `torch.normal(mu, std)` samples the same distribution and **breaks training**: the
> tensor it returns carries no gradient back to `mu`, so the encoder would never
> learn. That is exactly the problem 17.25 exists to solve. The check differentiates
> `z*` with respect to `mu` and rejects a detached sample.

**`forward`** returns the triple `(x_hat, mu, logvar)`. `train_vae` unpacks all three.

### Part 2 — Construct the loss (25 marks)

Call `reconstruction_loss` and `kl_divergence`, add them, and return
**`(loss, recon, kl)`**.

The triple matters: `train_vae` logs the two components separately, and that is what
gives `plot_metrics` its two panels. Returning the total alone fails the check.

The KL is **added**, not subtracted. Chapter 17 maximises the ELBO; here the negative
ELBO is minimised, so the penalty is positive. Getting the sign wrong makes the loss
rise, and the Part 3 check says so in those words.

Do not reimplement either term — the check compares your values against the provided
functions.

### Part 3 — Train it (25 marks)

```python
history = train_vae(vae, vae_loss, train_set)
plot_metrics(history)
```

`train_vae` builds the loader, wraps your model and loss in a LightningModule, runs
twelve epochs, and returns one row per epoch. Metrics go to `logs/metrics.csv`, which
is **overwritten on every run** — there is no version-numbered pile of directories to
sift through.

**The answer cell** asks which term rose. One of them does, monotonically, from the
first epoch to the last. Say why the model accepts that and what the rise is buying.

### Part 4 — Decode the latent sweep (20 marks)

`load_z_samples()`, `decode_z`, `show_grid`. Three lines.

The interesting part is what is already decided for you inside `z_samples.pt`: the
coordinates are even **in probability**, not in $z$. The prior is
$\text{Norm}_z[0,I]$, so the ten values on each axis are the midpoints of ten equal
probability slices mapped through the inverse normal CDF —
$\pm1.645, \pm1.036, \pm0.674, \pm0.385, \pm0.126$. A `linspace(-3, 3)` would spend
its outer cells in the tails, where the prior puts almost no mass and the model has
seen almost nothing, and those cells decode to smears. `make_z_samples.py` shows the
construction.

The final provided cell saves `vae.pt` for the bonus notebook.

---

## The bonus assignment (20 marks)

`assignment3_bonus.ipynb`. It loads `vae.pt` and trains nothing, so **finish the basic
assignment first** and keep both notebooks in the same directory.

| | | Marks |
|---|---|---|
| B1 | encode the test set, scatter the ten classes on the latent plane | 6 |
| B2 | the ten class centroids, and 50 points evenly spaced along $c_0 \to c_1 \to \cdots \to c_9$ | 6 |
| B3 | decode the 50 points | 4 |
| B4 | write `latent_walk.gif` with `imageio.v3`, plus the answer cell | 4 |

B2 is the only part with any real work in it. Spacing the 50 points evenly **along the
path** rather than evenly per segment keeps the animation at constant speed, and
`np.interp` against the cumulative segment lengths does it in four lines — the
notebook gives the recipe.

The payoff is that B1's scatter explains Part 4's grid. A cell of the grid that drew
nothing recognisable is a cell sitting in a gap between two clouds, and the animation
in B4 crosses several of those gaps on its way from `0` to `9`.

---

## Marking

| Part | | Marks |
|---|---|---|
| 1 | `VAE` with `self.encoder` and `self.decoder`, a differentiable `reparameterize`, a three-value `forward` | 30 |
| 2 | `vae_loss` returning `(loss, recon, kl)`, built from the provided terms | 25 |
| 3 | the training run and the two-panel plot, plus the answer cell | 25 |
| 4 | the latent sweep loaded, decoded and displayed, plus the answer cell | 20 |
| | **Basic total** | **100** |
| B1–B4 | the bonus notebook | **20 bonus** |

> **Do not clear the outputs before submitting.** The figures are the evidence; a
> notebook with cleared outputs is marked on its code alone and will lose most of the
> marks for Parts 3 and 4.

The **Your answer** cells carry the marks of the parts they close. An answer that only
restates what the figure shows earns none of them — say what it means.

**On the numbers you should expect.** After twelve epochs a correct run lands near
`recon` 150 and `kl` 7, a total around 157. You are **not** marked on hitting those
figures: training is stochastic and your hardware is not the marker's. You are marked
on having assembled the model correctly, trained it, and read your own output
honestly. A run that came out at 162 and is reported accurately beats one that claims
150 with no figure to show for it.

---

## Common ways this goes wrong

| symptom | cause |
|---|---|
| `the model has no attribute 'encoder'` | the sub-modules were given other names; `train.py` requires `self.encoder` and `self.decoder` |
| `z* has no grad_fn` | `torch.normal(mu, std)` instead of `mu + std * torch.randn_like(mu)` |
| `loss_fn must return the triple` | `vae_loss` returned only the total |
| the loss rises instead of falling | the KL was subtracted; you minimise the **negative** ELBO, so it is added |
| `recon is X, but components.reconstruction_loss gives Y` | the term was reimplemented instead of called |
| the KL falls to near zero and every cell of the grid is the same digit | posterior collapse — check the sign again |
| `vae.pt not found` in the bonus notebook | the basic notebook was not run to its last cell |
| `z_samples.pt not found` | re-run the Colab setup cell |
| training takes 20 minutes | the runtime is on CPU; switch to a T4 and re-run from the top |

---

## If you want to go further

Nothing below is marked.

- **Change `SIGMA2` in `components.py`.** It is the exchange rate between the two
  terms. At `1.0` the KL dominates, the posterior is pulled hard toward the prior, and
  the digits blur. At `0.01` reconstruction dominates, the digits sharpen, and the
  latent cloud spreads well outside the prior — so the fixed sweep in `z_samples.pt`
  covers less and less of it. Re-run Part 4 after each change and watch the grid and
  the bonus scatter move in opposite directions.
- **Raise `D_Z` to 10.** The loss will improve substantially and Part 4 stops being
  possible: you cannot sweep a ten-dimensional grid. This is why the assignment fixes
  $D_z = 2$, and it is worth seeing the trade directly.
- **Walk between two single images** rather than two centroids. Encode two test
  digits, interpolate between their $\mu$, and animate. Compare it with the centroid
  walk: individual digits sit further out than the centroids, so the path crosses more
  empty space and the morph is rougher.
- **Implement the $\beta$-VAE.** Multiply the KL term by $\beta > 1$ in `vae_loss` and
  look at what happens to the bonus scatter plot. The notes' disentanglement section
  says what to expect.
