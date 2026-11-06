# LAB4 — Tokens, Masks and an Encoder Transformer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kenpu-uoit/csci4052u-machine-learning-2/blob/main/labs/lab4/lab4.ipynb)

**Units.** `chapter_12_transformers`
**Duration.** 1 hour

Download BERT through the Hugging Face API, turn a string into sub-word tokens and embedding vectors, hide tokens behind [MASK] and measure how often the model recovers them from context alone, then pull out the attention matrix and find the head that routes the masked position onto the word that gives the answer away.

## Learning outcomes

1. **`prepare-text-for-transformers`** — Turn raw text into the input a transformer consumes, using sub-word tokenization and learned embeddings.
2. **`contrast-transformer-paradigms`** — Place BERT in the encoder family and run the masked language modelling objective it was trained with.
3. **`compute-dot-product-self-attention`** — Read self-attention as dynamic routing: an N x N matrix of non-negative weights whose rows sum to one.

## Exercises

| # | Exercise | Time | What you do |
|---|---|---|---|
| 1 | From a string to what the transformer sees | 15 min | Tokenize a sentence into sub-word pieces and indices, then look up the embedding matrix and embed the sentence with it. |
| 2 | Masking, and predicting the missing token | 30 min | Mask one token and read the model's five best guesses, then mask 15% of every sentence in the corpus and measure top-1 and top-5 accuracy. |
| 3 | The attention matrix | 10 min | Stack the per-layer attention tensors, verify the rows are distributions, and find the head that connects the mask to the word that settles it. |

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
| `lab4.ipynb` | the notebook you work in |
| `lab4_utils.py` | support code: datasets, model skeletons, plotting, and the checks |
