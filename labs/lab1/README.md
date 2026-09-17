# LAB1 — Shallow Networks, Losses, and Generalization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kenpu-uoit/csci4052u-machine-learning-2/blob/main/labs/lab1/lab1.ipynb)

**Units.** `preliminaries_to_machine_learning`, `training_models`
**Duration.** 1 hour

Build a shallow ReLU network from arithmetic and find its joints, derive least squares from the Gaussian negative log-likelihood and change one distribution to get cross-entropy, then measure how capacity trades bias against variance.

## Learning outcomes

1. **`construct-shallow-network`** — Construct a shallow ReLU network and trace how it produces a piecewise linear function.
2. **`derive-loss-from-likelihood`** — Derive a loss by having the network predict the parameters of a distribution over outputs.
3. **`decompose-test-error`** — Separate train from test error and say which of noise, bias, and variance dominates.

## Exercises

| # | Exercise | Time | What you do |
|---|---|---|---|
| 1 | A shallow ReLU network, by hand | 15 min | Implement the three-unit network with plain tensor arithmetic, locate its joints, and check a region's slope against the units active there. |
| 2 | The loss recipe | 25 min | Write the Gaussian negative log-likelihood, fit the regression model with Lightning, then change only the distribution and fit MNIST-1D. |
| 3 | Capacity, train error, and test error | 15 min | Sweep the hidden width on a small noisy training set and read bias, variance, and the noise floor off the resulting curves. |

## Getting started

Click the badge above to open the notebook in Colab, then **File → Save a copy in
Drive** before you start, so your work survives the session. Run the first two cells,
then work downward: each **YOUR CODE** cell is followed by a **CHECK** cell that tells
you whether you got it right.

To run it on your own machine instead, see [../README.md](../README.md).

## What to submit

The executed notebook, with every check printing `[ok]` and the written answers filled
in.

## Files

| File | |
|---|---|
| `lab1.ipynb` | the notebook you work in |
| `lab1_utils.py` | support code: datasets, model skeletons, plotting, and the checks |
