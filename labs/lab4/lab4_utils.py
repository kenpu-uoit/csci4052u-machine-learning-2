"""Support code for Lab 4 — tokens, masks and an encoder transformer.

Students import from this module and never edit it.  It carries the sentence
corpus, the BERT download, the seeded masking helper, the two plotting
helpers, and the ``check_*`` functions.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

DATA = Path(__file__).parent / "data"
# Cache the weights beside the lab so a second run needs no network.  This has
# to be set before transformers is imported.
os.environ.setdefault("HF_HOME", str(DATA / "hf"))

import matplotlib.pyplot as plt
import torch

MODEL_NAME = "bert-base-uncased"

# Measured constants.  They appear in the checks' messages so a student who
# misses one is told the number that was expected.
N_LAYERS = 12
N_HEADS = 12
HIDDEN = 768
VOCAB = 30522
MASK_RATE = 0.15
N_MASKED = 60          # masked positions over the 30 sentences at 15%
N_DEMO_TOKENS = 17     # tokens of DEMO, including [CLS] and [SEP]
N_FACT_TOKENS = 18     # tokens of DEMO_FACT, including [CLS] and [SEP]

KEY_COLOUR = "#0d6efd"
ALT_COLOUR = "#c2412d"

# The sentence whose tokenization shows sub-word splitting.
DEMO = "Tokenization splits unbelievably long words into subword pieces."

# The sentence Exercises 2 and 3 work on.  "Paris" is pinned by "France" two
# words earlier, which is what makes the masked prediction confident and the
# attention from the mask readable.
DEMO_FACT = "The capital city of France is Paris, and it sits on the river Seine."

# The sentence whose context fixes a category but not a word.
DEMO_VAGUE = "The restaurant on the corner serves the best pizza in the whole city."

SENTENCES = [
    "The capital city of France is Paris, and it sits on the river Seine.",
    "Water boils at one hundred degrees Celsius at sea level pressure.",
    "She opened the front door and stepped out into the cold morning air.",
    "The doctor asked the patient to describe the pain in her left shoulder.",
    "A neural network is trained by minimising a loss function with gradient descent.",
    "The train to the airport leaves from platform nine every twenty minutes.",
    "He poured the coffee into a cup and added two spoons of sugar.",
    "Most of the students finished the examination before the hour was over.",
    "The museum is closed on Mondays but open every other day of the week.",
    "Scientists have observed that the ice sheets are melting faster than expected.",
    "My brother plays the guitar and my sister prefers the piano.",
    "The company announced that it would hire two hundred new employees this year.",
    "Rain fell steadily all afternoon and the streets were soon covered in water.",
    "The committee voted to approve the budget after a long and difficult debate.",
    "A good dictionary will tell you both the meaning and the origin of a word.",
    "The cat jumped onto the table and knocked over a glass of milk.",
    "Students who study regularly tend to perform better on their final examinations.",
    "The bridge was built in the nineteenth century and is still in use today.",
    "He read the letter twice before he understood what his father had written.",
    "The restaurant on the corner serves the best pizza in the whole city.",
    "Many birds migrate south in the autumn and return again in the spring.",
    "The teacher explained the problem again and then asked the class a question.",
    "Her first novel was published when she was only twenty three years old.",
    "The engine would not start because the battery was completely flat.",
    "Light travels much faster than sound, which is why we see lightning first.",
    "The farmer sold his apples at the market every Saturday morning.",
    "A computer stores information as a long sequence of zeros and ones.",
    "They walked along the beach until the sun disappeared below the horizon.",
    "The hotel room was small but clean and the view from the window was lovely.",
    "Every child in the village learned to swim in the river during the summer.",
]


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------
def load_bert(name: str = MODEL_NAME):
    """Download (or load from cache) the BERT tokenizer and masked LM.

    ``attn_implementation="eager"`` is not optional: the default SDPA kernel
    ignores ``output_attentions=True``, and Exercise 3 needs the attention
    matrices.
    """
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    # transformers 5 reports the unused next-sentence-prediction head as
    # UNEXPECTED.  It is expected here; quiet it so the setup cell does not
    # open with four lines that look like errors.
    logging.getLogger("transformers").setLevel(logging.ERROR)

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForMaskedLM.from_pretrained(name, attn_implementation="eager")
    model.eval()
    return tokenizer, model


def describe(tokenizer, model) -> None:
    """Print the model's shape.  Called from the setup cell."""
    cfg = model.config
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{MODEL_NAME}: {n_params / 1e6:.0f}M parameters")
    print(f"  layers      {cfg.num_hidden_layers}")
    print(f"  heads/layer {cfg.num_attention_heads}")
    print(f"  hidden size {cfg.hidden_size}")
    print(f"  vocabulary  {tokenizer.vocab_size}")
    print(f"  mask token  {tokenizer.mask_token!r} = id {tokenizer.mask_token_id}")


def mask_positions(n_tokens: int, rate: float = MASK_RATE, seed: int = 0) -> list[int]:
    """Which positions of a sequence of ``n_tokens`` to mask.

    Positions 0 and ``n_tokens - 1`` are [CLS] and [SEP] and are never chosen.
    Seeded from ``seed`` plus the length, so every student masks the same
    positions and the accuracies in Exercise 2b are exact numbers.
    """
    body = list(range(1, n_tokens - 1))
    k = max(1, round(rate * len(body)))
    g = torch.Generator().manual_seed(seed + n_tokens)
    order = torch.randperm(len(body), generator=g)[:k]
    return sorted(body[i] for i in order.tolist())


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
def show_predictions(panels):
    """Bar charts of top-k predictions.

    ``panels`` is a list of ``(title, tokens, probabilities)``, one per axis.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 3.2))
    axes = [axes] if len(panels) == 1 else list(axes)
    for ax, (title, tokens, probs) in zip(axes, panels):
        y = range(len(tokens))
        ax.barh(list(y), list(probs), color=KEY_COLOUR)
        ax.set_yticks(list(y))
        ax.set_yticklabels(list(tokens))
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("probability")
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    plt.show()
    return fig


def show_attention(weights, tokens, title=""):
    """Heatmap of one head's N x N attention matrix."""
    w = weights.detach().cpu().numpy() if torch.is_tensor(weights) else weights
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(w, cmap="Blues", vmin=0.0)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("attended to (keys)")
    ax.set_ylabel("attending from (queries)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    plt.show()
    return fig


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
class CheckFailed(AssertionError):
    pass


def check(condition, message):
    if not condition:
        raise CheckFailed(message)


def _call(fn, *args, name="the function"):
    try:
        return fn(*args)
    except NotImplementedError:
        raise CheckFailed(f"{name} is not implemented yet.")


def check_tokenize(fn, tokenizer):
    """Exercise 1a."""
    pairs = _call(fn, DEMO, name="tokenize")
    check(
        pairs is not None,
        "tokenize returned None -- the function has to return the list, not print it.",
    )
    pairs = list(pairs)
    check(
        len(pairs) == N_DEMO_TOKENS,
        f"{len(pairs)} token pairs for DEMO, expected {N_DEMO_TOKENS} -- "
        "did you keep the special tokens? The tokenizer adds [CLS] and [SEP] by default.",
    )
    for p in pairs:
        check(
            len(p) == 2,
            f"each entry should be a (token, id) pair; got {p!r}.",
        )
    tokens = [t for t, _ in pairs]
    ids = [int(i) for _, i in pairs]
    check(
        tokens[0] == "[CLS]" and tokens[-1] == "[SEP]",
        f"the sequence runs {tokens[0]!r} ... {tokens[-1]!r}, expected "
        "'[CLS]' ... '[SEP]' -- these two frame every BERT input.",
    )
    check(
        tokenizer.convert_tokens_to_ids(tokens) == ids,
        "the ids do not match the tokens -- convert_tokens_to_ids(tokens) disagrees "
        "with the second half of your pairs.",
    )
    pieces = [t for t in tokens if t.startswith("##")]
    for want in ("##ization", "##bly"):
        check(
            want in tokens,
            f"{want!r} is not among the tokens -- expected sub-word splitting of "
            "'Tokenization' and 'unbelievably'. Did you call tokenizer.tokenize on "
            "pre-split words?",
        )
    print(f"[ok] 1a: {len(pairs)} tokens, {len(pieces)} of them '##' continuation pieces: "
          f"{pieces}")
    print("     Sub-word tokenization: a word the vocabulary does not hold is spelled "
          "out of pieces it does.")


def check_embeddings(matrix, embedded, pairs):
    """Exercise 1b."""
    check(
        hasattr(matrix, "shape"),
        "the first value should be the embedding WEIGHT matrix (a tensor), not the "
        "nn.Embedding module -- take .weight.",
    )
    check(
        tuple(matrix.shape) == (VOCAB, HIDDEN),
        f"the embedding matrix has shape {tuple(matrix.shape)}, expected "
        f"({VOCAB}, {HIDDEN}) -- one row per vocabulary entry, each of the hidden size.",
    )
    check(
        tuple(embedded.shape) == (N_DEMO_TOKENS, HIDDEN),
        f"the embedded sentence has shape {tuple(embedded.shape)}, expected "
        f"({N_DEMO_TOKENS}, {HIDDEN}) -- one vector per token, batch dimension removed.",
    )
    ids = torch.tensor([int(i) for _, i in pairs])
    want = matrix[ids]
    check(
        torch.equal(embedded.detach().float(), want.detach().float()),
        "the embedded rows are not the embedding matrix's rows for those ids -- "
        "embedding is a lookup, so row n must equal matrix[id_n] exactly.",
    )
    print(f"[ok] 1b: embedding matrix {tuple(matrix.shape)}, "
          f"DEMO embedded to {tuple(embedded.shape)}.")
    print(f"     {VOCAB} x {HIDDEN} is {VOCAB * HIDDEN / 1e6:.1f}M parameters before any "
          "transformer layer -- which is why the vocabulary size is a design choice.")


def check_one_mask(tokens, probs, position_token="paris"):
    """Exercise 2a."""
    tokens, probs = list(tokens), [float(p) for p in probs]
    check(
        len(tokens) == 5 and len(probs) == 5,
        f"got {len(tokens)} tokens and {len(probs)} probabilities, expected 5 of each "
        "-- use topk(5) on the softmax of the logits at the masked position.",
    )
    check(
        all(0.0 <= p <= 1.0 for p in probs),
        f"the values {probs} are not all in [0, 1] -- these should be probabilities, "
        "so apply softmax to the logits before topk.",
    )
    check(
        probs == sorted(probs, reverse=True),
        f"the probabilities {probs} are not descending -- topk returns them sorted; "
        "did the tokens and values get paired up in different orders?",
    )
    check(
        tokens[0] == position_token,
        f"the top prediction is {tokens[0]!r}, expected {position_token!r} -- "
        "is the mask at the right position, and are you reading the logits at that "
        "same position?",
    )
    check(
        probs[0] > 0.80,
        f"the top probability is {probs[0]:.3f}, expected above 0.80 -- a low value "
        "here usually means the logits were read at the wrong position.",
    )
    print(f"[ok] 2a: {tokens[0]!r} at p={probs[0]:.3f}; "
          f"runners-up {list(zip(tokens[1:], [round(p, 3) for p in probs[1:]]))}")
    print("     The masked position carries no information about its own token, so every "
          "bit of that came from the other positions. That is the encoder's bidirectional "
          "context.")


def check_accuracy(n_masked, top1, top5):
    """Exercise 2b."""
    check(
        n_masked == N_MASKED,
        f"{n_masked} positions masked in total, expected {N_MASKED} -- "
        "mask_positions(len(ids)) gives the indices for one sentence; sum over all 30.",
    )
    for name, value in (("top-1", top1), ("top-5", top5)):
        check(
            0.0 <= value <= 1.0,
            f"{name} accuracy came out at {value} -- that is a count, not a fraction; "
            f"divide by the {N_MASKED} masked positions.",
        )
    check(
        top5 >= top1,
        f"top-5 ({top5:.3f}) is below top-1 ({top1:.3f}) -- top-5 counts a hit anywhere "
        "in the five, so it can never be smaller.",
    )
    check(
        0.40 <= top1 <= 0.75,
        f"top-1 accuracy {top1:.3f} is outside [0.40, 0.75] -- is the gold token read "
        "from the UNMASKED ids, and the prediction from the masked ones?",
    )
    check(
        0.60 <= top5 <= 0.95,
        f"top-5 accuracy {top5:.3f} is outside [0.60, 0.95].",
    )
    print(f"[ok] 2b: {n_masked} masked positions, top-1 {top1:.3f}, top-5 {top5:.3f}")
    print(f"     {MASK_RATE:.0%} is the rate BERT was pre-trained at. More than half the "
          "hidden tokens come back exactly, from context alone, and three quarters are "
          "in the model's top five.")


def check_prediction_figure(fig, p_constrained, p_vague):
    """Exercise 2c."""
    axes = fig.axes if fig is not None else []
    check(
        len(axes) == 2,
        f"the figure has {len(axes)} axes, expected 2 -- pass both panels to "
        "show_predictions in one call.",
    )
    for ax in axes:
        check(
            len(ax.patches) == 5,
            f"an axis has {len(ax.patches)} bars, expected 5.",
        )
    check(
        p_constrained > 2 * p_vague,
        f"the constrained top probability {p_constrained:.3f} is not twice the "
        f"unconstrained one {p_vague:.3f} -- are the two masks in the sentences you "
        "think they are?",
    )
    print(f"[ok] 2c: constrained {p_constrained:.3f} against unconstrained "
          f"{p_vague:.3f}, a factor of {p_constrained / p_vague:.1f}")
    print("     The width of the softmax is the model's uncertainty, and it is a "
          "property of the context, not of the word.")


def check_attention(stack):
    """Exercise 3a."""
    check(
        torch.is_tensor(stack),
        f"expected one stacked tensor, got {type(stack).__name__} -- "
        "torch.stack(outputs.attentions) turns the tuple of 12 into one array. "
        "If the tuple is empty, the model was not loaded with "
        'attn_implementation="eager".',
    )
    want = (N_LAYERS, N_HEADS, N_FACT_TOKENS, N_FACT_TOKENS)
    check(
        tuple(stack.shape) == want,
        f"the stack has shape {tuple(stack.shape)}, expected {want} -- "
        "(layers, heads, N, N). Each per-layer tensor arrives with a leading batch "
        "dimension of 1; squeeze it.",
    )
    check(
        bool((stack >= 0).all()),
        f"the smallest entry is {stack.min().item():.3g} -- attention weights come out "
        "of a softmax and cannot be negative. Are these the weights, or the scores "
        "before the softmax?",
    )
    sums = stack.sum(-1)
    check(
        bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-5)),
        f"row sums range over [{sums.min():.4f}, {sums.max():.4f}], expected 1 -- "
        "the softmax is over the LAST dimension, so each row is a distribution over "
        "the tokens attended to.",
    )
    print(f"[ok] 3a: attention stack {tuple(stack.shape)}, all entries >= 0, "
          "every row sums to 1.")
    print(f"     {N_LAYERS} x {N_HEADS} = {N_LAYERS * N_HEADS} matrices, each "
          f"{N_FACT_TOKENS} x {N_FACT_TOKENS}. Both of the last two dimensions are N, "
          "which is where the O(N^2) cost of self-attention comes from.")


def check_routing(layer, head, weight, fig):
    """Exercise 3b."""
    check(
        weight > 0.40,
        f"the strongest mask -> 'france' weight is {weight:.3f}, expected above 0.40 -- "
        "search all layers and all heads, and read the row of the MASKED position "
        "(the query) at the column of 'france' (the key).",
    )
    check(
        layer >= 7,
        f"the winning head is in layer {layer}; expected layer 7 or deeper -- "
        "if it came out in layer 0 or 11 the two indices may be the wrong way round, "
        "or the row and column swapped.",
    )
    check(
        fig is not None and len(fig.axes) >= 1,
        "no heatmap was drawn -- pass the chosen head's matrix to show_attention.",
    )
    print(f"[ok] 3b: layer {layer}, head {head} routes {weight:.3f} of the masked "
          "position's attention onto 'france'.")
    print("     That connection is not in the weights. It was computed from this "
          "sentence, which is what 'dynamic routing' means in the notes.")
