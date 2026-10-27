# Assignment 2 — Fine-tuning YOLOv3 on a New Dataset

**CSCI 4052U — Machine Learning II**
**Unit.** `fast_object_detection_networks` · **Builds on.** Lab 3
**Weight.** 100 marks · **Expected effort.** 2–3 hours, most of it reading output

---

## The question this assignment asks

Lab 3 took a COCO-pretrained YOLOv3-tiny exactly as it came and measured where it
failed. Every class it could name was one of COCO's 80.

Real detection problems are not COCO problems. You have a few hundred images of
something COCO never heard of, and a detector that was trained on something else. This
assignment walks the standard answer: **start from the pretrained weights and fine-tune
on your own data.**

The dataset is **African Wildlife** — four classes, `buffalo`, `elephant`, `rhino`,
`zebra`. The choice is not arbitrary:

| class | in COCO's 80? |
|---|---|
| `elephant` | yes |
| `zebra` | yes |
| `buffalo` | **no** |
| `rhino` | **no** |

So the pretrained detector arrives already able to do half the task and unable to do the
other half. Watching what fine-tuning does to each half is the point of the assignment.

---

## What you will produce

One file: **`assignment2.ipynb`, run from top to bottom, with its outputs intact.**

There is no separate report. The four markdown cells headed **Your answer** are where
you write, two or three sentences each. Everything else is code and the figures and
tables it prints.

> **Do not clear the outputs before submitting.** The figures and tables are the
> evidence; a notebook with cleared outputs is marked on its code alone and will lose
> most of the marks for Parts 1 and 3.

---

## Before you start

**Turn on a GPU.** On Colab: *Runtime → Change runtime type → T4 GPU*. On the free tier
this is enough — the whole assignment trains two small models and uses well under 4 GB
of GPU memory.

**Save a copy first.** *File → Save a copy in Drive*. Colab discards everything when the
session ends.

**Expected timings on a Colab T4**, measured:

| step | time |
|---|---|
| installing `ultralytics` | ~1 min, once |
| downloading the dataset (~100 MB) | ~1 min, once |
| Part 1.2, inference over 227 test images | under a minute |
| Part 2.1, fine-tuning 10 epochs at 416 px | 3–5 min |
| Part 4, one more training run | 3–6 min |

If a cell is taking far longer than this, check that `DEVICE` printed a GPU name rather
than `CPU` in the setup cell.

---

## How to do it, part by part

### Part 1 — What the pretrained detector already knows (20 marks)

**1.1 (8 marks)** — Load `yolov3-tinyu.pt`, predict on the first 8 test images, display
with the provided `show(...)` helper.

The one thing to get right is the **titles**: each panel must say the image's *true*
class, so a reader can see where the model is wrong. `ground_truth_classes(p)` gives you
the class indices and `CLASSES` maps those to names. An untitled grid of pictures earns
about half.

**1.2 (12 marks)** — The counting exercise. For every test image annotated with exactly
one class, run the model and tally what COCO labels it produced.

- Skip images with more than one class — `if len(gt) != 1: continue`. They muddle the
  tally and the instruction says to skip them.
- The model's name for class index `c` is `pretrained.names[int(c)]`, which indexes
  **COCO's** 80 names. Do not use `CLASSES` here; that is the wildlife mapping and the
  pretrained model knows nothing about it.
- A `collections.Counter` per wildlife class, then `.most_common(3)`, is all the
  machinery needed.

This runs one image at a time through a loop. That is fine at this size — it is under a
minute.

**1.3 (the answer cell)** — You are expected to notice that `elephant` and `zebra` come
out roughly right, and that `buffalo` and `rhino` get labelled as whatever COCO animal
is nearest. Say *which* COCO labels they attract, and connect that to the table above.
Simply writing "the model performs poorly on some classes" earns nothing.

### Part 2 — Fine-tune it (30 marks)

**2.1 (20 marks)** — A single `model.train(...)` call with the settings given in the
table. Start from a **fresh** `YOLO("yolov3-tinyu.pt")`, not the object you used in
Part 1, so the run starts from the COCO weights rather than from wherever Part 1 left
things.

Ultralytics handles the part that would otherwise be fiddly: it sees that the dataset
has 4 classes where the checkpoint has 80, builds a new detection head of the right
size, and keeps the pretrained backbone underneath. That is the whole mechanism of
transfer learning for detection, and it is one argument.

Keep the returned object — `train_results.save_dir` is where 2.2 reads from.

**2.2 (10 marks)** — Read `results.csv` from that directory and plot `metrics/mAP50(B)`
against `epoch`.

The header row of that file has **padding spaces** in the column names, which is why the
`{k.strip(): v for k, v in r.items()}` line is given to you. Without it, looking up
`"metrics/mAP50(B)"` raises a `KeyError`. Every value also arrives as a string, so wrap
it in `float(...)` before plotting.

Label both axes. An unlabelled plot loses marks.

**2.3 (the answer cell)** — Look at your own curve before writing. It is normal for the
first few epochs to wobble, including downward, before the climb. Say whether the curve
had flattened by epoch 10 or was still rising, and point at what in the plot shows it.

### Part 3 — Measure what changed (30 marks)

**3.1 (12 marks)** — Evaluate on the **test** split, not validation. The training loop
has been selecting checkpoints against the validation split all along, so validation
numbers are mildly optimistic. `split="test"` is the argument that matters.

Report both `metrics.box.map50` (mAP at IoU 0.5) and `metrics.box.map` (mAP averaged
over IoU 0.5 to 0.95, the stricter COCO-style number). Lab 3 §2c is where these were
introduced.

**3.2 (8 marks)** — One row per class. `metrics.box.ap50` is an array in the order of
`metrics.names`, so `zip` the two.

**3.3 (10 marks)** — Re-run the **same 8 images** from 1.1 through the fine-tuned model
and show them with the same titles. Using different images defeats the comparison and
loses most of the marks for this part.

**3.4 (the answer cell)** — The interesting question is whether the two classes COCO
already knew ended up better than the two it did not. Look at your per-class table and
answer from it. Then say what that implies about *which part of the network* the
fine-tuning had to change — the convolutional features that were already good at
animals, or the head that names them.

### Part 4 — One experiment of your own (20 marks)

Pick **one** of A, B, or C. Not two, not all three.

| | change | the question it answers |
|---|---|---|
| **A** | `epochs=5` | how much of the benefit arrives in the first few epochs |
| **B** | `imgsz=640` | whether a larger input helps when the objects are already large |
| **C** | `freeze=10` | whether the COCO backbone needs to change at all |

Three rules, and the marks hang on them:

1. **Change one thing.** Everything else identical, `seed=0` included. Two changes at
   once and you cannot attribute the difference to either.
2. **Use a different `name=`.** `name="experiment"` keeps the Part 2 run intact. Reusing
   `name="finetune"` with `exist_ok=True` overwrites it and you lose your baseline.
3. **Evaluate the same way** — test split, and at the **same `imgsz` you trained with**.
   Option B in particular must be validated at 640, not 416.

Then fill in the two-row table and write three sentences: what you changed, what
happened to accuracy, what happened to time, and whether it was worth it.

> A result where your change made things *worse*, or changed nothing, is a perfectly
> good answer and earns full marks if you report it honestly and say what you conclude.
> Do not re-run with a different seed until you get the answer you expected.

---

## Marking

| Part | | Marks |
|---|---|---|
| 1.1 | pretrained predictions displayed, titled with the true class | 8 |
| 1.2 | confusion counts over the test split | 12 |
| 2.1 | fine-tuning run with the stated settings | 20 |
| 2.2 | mAP50 against epoch, axes labelled | 10 |
| 3.1 | test-split mAP50 and mAP50-95 | 12 |
| 3.2 | per-class mAP50 table | 8 |
| 3.3 | the same 8 images after fine-tuning | 10 |
| 4 | one changed setting, re-trained, table filled | 20 |
| | **Total** | **100** |

The **Your answer** cells carry the marks of the parts they close. An answer that only
restates the numbers earns none of them — say what the numbers mean.

**On the numbers you should expect.** A correct Part 2 lands around mAP50 0.90–0.95 on
the test split. You are **not** marked on hitting a particular figure: training is
stochastic and your hardware is not the marker's. You are marked on having run the
thing, measured it properly, and read your own output honestly. A run that came out at
0.88 and is reported accurately beats one that claims 0.95 with no output to show for
it.

---

## Common ways this goes wrong

| symptom | cause |
|---|---|
| `KeyError: 'metrics/mAP50(B)'` | the `results.csv` header has padding spaces — strip the key names |
| Training takes 40 minutes | the runtime is on CPU; switch to a T4 and re-run from setup |
| Part 4 numbers identical to Part 2 | the variant reused `name="finetune"` and read back the old run |
| Option B scores worse than the baseline | that is a real result — report it, do not hide it |
| `CUDA out of memory` | lower `batch` to 8; nothing else in the assignment needs changing |
| Part 1 shows nothing on some images | `conf=0.25` is a threshold, not a bug; images where the model is unsure are part of the finding |

---

## If you want to go further

Nothing below is marked.

- Evaluate the Part 2 model on **COCO128** from Lab 3. It will do badly, because its
  head now has four outputs instead of eighty. That is catastrophic forgetting, and it
  is the price of fine-tuning rather than adding a head.
- Try `yolov8n.pt` in place of `yolov3-tinyu.pt`, changing nothing else, and see what six
  years of architecture work is worth on a dataset this size.
- Re-run Part 2 with `pretrained=False` to train from random initialisation. The gap
  between that and your fine-tuned run is what the ImageNet-and-COCO pre-training was
  worth, and it is the measurement the R-CNN paper made in 2014.
