# Laboratories

| Lab | Topic | Units | Open |
|---|---|---|---|
| [LAB1](./lab1/) | Shallow Networks, Losses, and Generalization | `preliminaries_to_machine_learning`, `training_models` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kenpu-uoit/csci4052u-machine-learning-2/blob/main/labs/lab1/lab1.ipynb) |

## Running a lab on Colab

Click the lab's badge above, then **Runtime → Run all**. The notebook's first cell
installs the two packages Colab does not already carry and downloads the lab's support
module; it takes a few seconds.

Two things worth knowing:

- **No GPU is needed.** These models are small enough that a GPU would spend more time
  moving data than computing, so the labs pin themselves to the CPU. Leaving the runtime
  on CPU also keeps everyone's numbers identical.
- **Colab forgets everything when the session ends.** Save your work with
  **File → Save a copy in Drive** as soon as you open the notebook, and work in that
  copy. Downloaded datasets are re-fetched automatically each session.

## Running a lab on your own machine

```bash
git clone https://github.com/kenpu-uoit/csci4052u-machine-learning-2.git
cd csci4052u-machine-learning-2/labs
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Then open `labN/labN.ipynb`. Running locally, the notebook's first cell does nothing —
`labN_utils.py` is already beside the notebook.

`requirements.txt` gives loose bounds rather than exact pins, because the right torch
build depends on your own CUDA version. See
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) if you want
GPU support; the CPU build is enough for every lab.

## What you submit

The executed notebook, with every check printing `[ok]` and the written answers filled
in. Your instructor will tell you where to hand it in.
