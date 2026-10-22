# LAB3 — Running a Single-Shot Detector, and Measuring Where It Fails

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kenpu-uoit/csci4052u-machine-learning-2/blob/main/labs/lab3/lab3.ipynb)

**Units.** `fast_object_detection_networks`
**Duration.** 1 hour

Run a pretrained YOLOv3-tiny over 32 annotated COCO images, match its boxes to the ground truth by IoU, and measure the error profile the unit describes: the labels are almost always right, the boxes are loose, and more than half the objects are missed. Then find the two places it fails -- the confidence threshold, and small objects.

## Learning outcomes

1. **`single-shot-formulation`** — Formulate detection as one regression from image pixels to boxes and class probabilities over a fixed grid.
2. **`yolo-error-profile`** — Diagnose a single-shot detector's error profile and its structural limits.
3. **`detector-design-tradeoffs`** — Choose a detector for a deployment constraint by reasoning about speed, accuracy and object scale.

## Exercises

| # | Exercise | Time | What you do |
|---|---|---|---|
| 1 | One forward pass, and what comes out of it | 15 min | Convert YOLO-format ground truth to pixel corners, then run the detector over 32 images and read the boxes, scores and labels it returns. |
| 2 | Matching, drawing, and the error profile | 25 min | Implement IoU and a greedy matcher, draw predictions against ground truth, then compute recall, mean IoU and label accuracy. |
| 3 | Where it fails | 15 min | Sweep the confidence threshold to see recall traded against localisation, then split the objects by area and measure the small-object gap. |

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
| `lab3.ipynb` | the notebook you work in |
| `lab3_utils.py` | support code: datasets, model skeletons, plotting, and the checks |
