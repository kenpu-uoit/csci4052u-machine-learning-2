# LAB2 — Depth, Degradation, and Residual Connections

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kenpu-uoit/csci4052u-machine-learning-2/blob/main/labs/lab2/lab2.ipynb)

**Units.** `convnets`
**Duration.** 1 hour

Build a convolutional block and a shallow convnet that trains easily, then a nine-layer one that does not -- measured on the training set, so the failure is optimisation rather than overfitting. Add skip connections to exactly the same nine convolutions and watch it train, then add batch normalization and measure the gradient reaching the first layer.

## Learning outcomes

1. **`extend-to-2d`** — Extend convolution to 2D images and read a standard image classification network.
2. **`diagnose-deep-network-training`** — Explain why stacking more layers eventually makes training worse, not better.
3. **`build-residual-blocks`** — Build a residual block and explain how skip connections restore gradient flow.

## Exercises

| # | Exercise | Time | What you do |
|---|---|---|---|
| 1 | A convolutional block, and a shallow convnet | 15 min | Write the 3x3 convolution, ReLU and max-pool block, count its parameters, stack three of them into a classifier and train it on MNIST. |
| 2 | Depth without help | 20 min | Build a nine-convolution network with the same widths and pooling, train it with identical settings, and compare the two on training error. |
| 3 | Residual connections, then batch normalization | 20 min | Write a residual block, rewire the same nine convolutions through it, add batch normalization, and measure the gradient norm at the first layer. |

## Getting started

Click the badge above to open the notebook in Colab, then **File → Save a copy in
Drive** before you start, so your work survives the session. Run the first two cells,
then work downward: each **YOUR CODE** cell is followed by a **CHECK** cell that tells
you whether you got it right.

To run it on your own machine instead, see [../README.md](../README.md).

## What to submit

The executed notebook, with every check printing `[ok]`.

## Files

| File | |
|---|---|
| `lab2.ipynb` | the notebook you work in |
| `lab2_utils.py` | support code: datasets, model skeletons, plotting, and the checks |
