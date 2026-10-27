# CSCI 4052U — Machine Learning II

Public course material for CSCI 4052U at Ontario Tech University: the laboratory
notebooks and the resources that go with them.

## Laboratories

Each lab is a self-contained Jupyter notebook that runs in Google Colab. Click the
badge, then **Runtime → Run all**. Nothing to install, and no GPU required.

| Lab | Topic | Duration | Open |
|---|---|---|---|
| [LAB1](labs/lab1/) | Shallow Networks, Losses, and Generalization | 1 hour | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kenpu-uoit/csci4052u-machine-learning-2/blob/main/labs/lab1/lab1.ipynb) |
| [LAB2](labs/lab2/) | Depth, Degradation, and Residual Connections | 1 hour | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kenpu-uoit/csci4052u-machine-learning-2/blob/main/labs/lab2/lab2.ipynb) |
| [LAB3](labs/lab3/) | Running a Single-Shot Detector, and Measuring Where It Fails | 1 hour | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kenpu-uoit/csci4052u-machine-learning-2/blob/main/labs/lab3/lab3.ipynb) |

See [labs/README.md](labs/README.md) for the local-install route and for how to save
your work.

## Assignments

Graded work. Each one has its own README with the instructions and the mark breakdown.

| Assignment | Topic | Weight | Open |
|---|---|---|---|
| [ASSIGNMENT2](assignments/assignment2/) | Fine-tuning YOLOv3 on a New Dataset | 100 marks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kenpu-uoit/csci4052u-machine-learning-2/blob/main/assignments/assignment2/assignment2.ipynb) |

Unlike the labs, an assignment may need a GPU runtime; its README says so.

## How the labs work

Every notebook follows the same three-cell rhythm:

- **SETUP** cells are provided. Run them and leave them alone.
- **YOUR CODE** cells are yours. Each contains `raise NotImplementedError()` — delete
  that line and write the code.
- **CHECK** cells verify what you wrote and print `[ok]` when it is right. When a check
  fails it tells you *what* is wrong, not merely *that* something is, so read the message
  before changing anything.

You can therefore tell whether your work is correct before you submit it.

## Contents

```
labs/          laboratory notebooks, one folder per lab
```

Lecture notes and slides are distributed separately through the course site.
