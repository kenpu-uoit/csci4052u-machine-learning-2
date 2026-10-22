"""Support code for Lab 3 — running a single-shot detector and measuring it.

Students import from this module and never edit it.  It carries the dataset
download, the model download, the ground-truth reader, the drawing helpers, a
reference matcher the checks lean on, and the ``check_*`` functions.
"""

from __future__ import annotations

import os
import time
import urllib.request
import zipfile
from pathlib import Path

os.environ.setdefault("YOLO_VERBOSE", "false")

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

DATA = Path(__file__).parent / "data"
COCO128_URL = "https://ultralytics.com/assets/coco128.zip"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov3-tinyu.pt"

# How many of the 128 images the lab uses, and the confidence the exercises
# are calibrated against.  Both appear in the checks' messages.
N_IMAGES = 32
CONF = 0.25
N_GT = 209          # ground-truth boxes over those 32 images
N_PRED = 144        # predictions at conf=0.25
N_MATCHED = 97      # matches at IoU >= 0.5
SMALL_AREA = 0.05   # fraction of image area below which a box counts as small

GT_COLOUR = "#0d6efd"
PR_COLOUR = "#c2412d"


# --------------------------------------------------------------------------
# data and model
# --------------------------------------------------------------------------
def download_coco128(root: Path = DATA) -> Path:
    """Fetch and unpack COCO128.  Cached; a second call does nothing."""
    root.mkdir(parents=True, exist_ok=True)
    out = root / "coco128"
    if out.exists():
        return out
    zip_path = root / "coco128.zip"
    if not zip_path.exists():
        t0 = time.time()
        urllib.request.urlretrieve(COCO128_URL, zip_path)
        print(f"downloaded coco128.zip ({zip_path.stat().st_size / 1e6:.1f} MB) "
              f"in {time.time() - t0:.1f}s")
    zipfile.ZipFile(zip_path).extractall(root)
    return out


def image_paths(n: int = N_IMAGES, root: Path = DATA) -> list[Path]:
    """The first ``n`` COCO128 images, in a fixed order."""
    folder = download_coco128(root) / "images" / "train2017"
    return sorted(folder.glob("*.jpg"))[:n]


def load_model(root: Path = DATA):
    """Load YOLOv3-tiny, downloading the weights on first use.

    Ultralytics publishes this checkpoint as ``yolov3-tinyu.pt``: the YOLOv3-tiny
    backbone with an anchor-free head retrofitted, not the 2018 network verbatim.
    For this lab it is a black box that turns an image into boxes, scores and
    labels in one forward pass.
    """
    from ultralytics import YOLO

    weights = root / "yolov3-tinyu.pt"
    if not weights.exists():
        root.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        urllib.request.urlretrieve(MODEL_URL, weights)
        print(f"downloaded yolov3-tinyu.pt ({weights.stat().st_size / 1e6:.1f} MB) "
              f"in {time.time() - t0:.1f}s")
    return YOLO(str(weights))


def load_ground_truth(path: Path) -> list[tuple]:
    """The raw annotation rows for one image, as stored.

    COCO128 keeps one text file per image, a row per object, in YOLO format:
    ``class cx cy w h`` with the four box numbers normalised to [0, 1].
    Converting those to pixel corners is Exercise 1a.
    """
    label = path.parent.parent.parent / "labels" / "train2017" / f"{path.stem}.txt"
    rows = []
    if label.exists():
        for line in label.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            rows.append((int(parts[0]), *(float(v) for v in parts[1:5])))
    return rows


def image_size(path: Path) -> tuple[int, int]:
    """(width, height) in pixels."""
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def class_names() -> dict[int, str]:
    """COCO class index to name, taken from the model itself."""
    return load_model().names


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------
def draw_boxes(ax, path, boxes, colour, names=None, title=None):
    """Draw ``(class, x1, y1, x2, y2[, score])`` boxes over an image."""
    from PIL import Image

    with Image.open(path) as im:
        ax.imshow(im.convert("RGB"))
    for box in boxes:
        cls, x1, y1, x2, y2 = box[:5]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               fill=False, edgecolor=colour, linewidth=2.0))
        label = names[int(cls)] if names else str(int(cls))
        if len(box) > 5:
            label = f"{label} {box[5]:.2f}"
        ax.text(x1, y1 - 4, label, fontsize=7, color="white",
                bbox=dict(facecolor=colour, edgecolor="none", pad=1.0))
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)


def show_comparison(paths, gts, preds, names, n=4):
    """Ground truth on the left, predictions on the right, ``n`` rows."""
    n = min(n, len(paths))
    fig, axes = plt.subplots(n, 2, figsize=(9, 3.1 * n))
    axes = axes.reshape(n, 2)
    for row in range(n):
        draw_boxes(axes[row][0], paths[row], gts[row], GT_COLOUR, names,
                   "ground truth" if row == 0 else None)
        draw_boxes(axes[row][1], paths[row], preds[row], PR_COLOUR, names,
                   "YOLOv3-tiny" if row == 0 else None)
    fig.tight_layout()
    plt.show()
    return fig


def sweep_table(rows):
    """Print the confidence sweep as a table."""
    print(f"{'conf':>6}{'boxes':>8}{'recall':>9}{'mean IoU':>11}{'label acc':>12}")
    for conf, n_pred, recall, miou, acc in rows:
        print(f"{conf:>6.2f}{n_pred:>8d}{recall:>9.1%}{miou:>11.3f}{acc:>12.1%}")


# --------------------------------------------------------------------------
# reference implementations, used only by the checks
# --------------------------------------------------------------------------
def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _match(gts, preds, thr=0.5):
    """Greedy: each ground-truth box takes the best unused prediction."""
    used, out = set(), []
    for gi, g in enumerate(gts):
        best, best_i = -1.0, -1
        for pi, p in enumerate(preds):
            if pi in used:
                continue
            v = _iou(g[1:5], p[1:5])
            if v > best:
                best, best_i = v, pi
        if best_i >= 0 and best >= thr:
            used.add(best_i)
            out.append((gi, best_i, best))
    return out


# --------------------------------------------------------------------------
# verification
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


def check_gt_to_xyxy(fn, paths):
    """Exercise 1a."""
    rows = [(2, 0.5, 0.5, 0.25, 0.5)]
    got = _call(fn, rows, 640, 480, name="gt_to_xyxy")
    check(len(got) == 1, f"gt_to_xyxy returned {len(got)} boxes for 1 row.")
    cls, x1, y1, x2, y2 = got[0]
    check(int(cls) == 2,
          f"the class came back as {cls}, expected 2 -- keep the class index as it is, "
          "it is not a coordinate.")
    want = (240.0, 120.0, 400.0, 360.0)
    for name, have, expect in zip("x1 y1 x2 y2".split(), (x1, y1, x2, y2), want):
        check(
            abs(have - expect) < 1e-6,
            f"on a 640x480 image, a box at cx=0.5 cy=0.5 w=0.25 h=0.5 should give "
            f"{name}={expect}, got {have} -- the stored numbers are the CENTRE and the "
            "width/height, normalised; corners are centre -/+ half the side.",
        )

    total, out_of_bounds = 0, 0
    for p in paths:
        w, h = image_size(p)
        boxes = fn(load_ground_truth(p), w, h)
        total += len(boxes)
        for _, bx1, by1, bx2, by2 in boxes:
            if bx1 < -1 or by1 < -1 or bx2 > w + 1 or by2 > h + 1 or bx2 <= bx1 or by2 <= by1:
                out_of_bounds += 1
    check(
        out_of_bounds == 0,
        f"{out_of_bounds} converted boxes fall outside their image or have a "
        "non-positive side -- did width and height get swapped?",
    )
    check(
        total == N_GT,
        f"the {len(paths)} images converted to {total} ground-truth boxes, expected {N_GT} "
        "-- every row of every label file should produce exactly one box.",
    )
    print(f"[ok] 1a: {total} ground-truth boxes, all inside their images.")


def check_predictions(predictions, paths, elapsed=None):
    """Exercise 1b."""
    check(
        len(predictions) == len(paths),
        f"predictions holds {len(predictions)} entries for {len(paths)} images -- "
        "one list per image, including images where nothing was found.",
    )
    flat = [b for per_image in predictions for b in per_image]
    check(
        len(flat) == N_PRED,
        f"{len(flat)} predicted boxes in total, expected {N_PRED} -- is conf set to "
        f"{CONF}, and is device='cpu'?",
    )
    for b in flat:
        check(
            len(b) == 6,
            f"a prediction has {len(b)} fields, expected 6: (class, x1, y1, x2, y2, score).",
        )
        cls, x1, y1, x2, y2, score = b
        check(0 <= int(cls) <= 79, f"class index {cls} is outside [0, 79].")
        check(CONF - 1e-6 <= score <= 1.0,
              f"a score of {score:.3f} is outside [{CONF}, 1] -- boxes below the "
              "confidence threshold should already have been dropped.")
        check(x2 > x1 and y2 > y1,
              f"a box has non-positive width or height: ({x1:.1f}, {y1:.1f}, "
              f"{x2:.1f}, {y2:.1f}) -- are the coordinates corners (xyxy) rather "
              "than centre+size?")
    print(f"[ok] 1b: {len(flat)} boxes from {len(paths)} images, every score >= {CONF}.")
    if elapsed is not None:
        print(f"     {elapsed / len(paths) * 1000:.0f} ms per image on CPU.")
    print("     One evaluation of one network produced every box in each image: no")
    print("     proposal stage, and the cost does not grow with the number of objects.")


def check_iou(fn):
    """Exercise 2a."""
    a = (10.0, 10.0, 30.0, 30.0)
    got = _call(fn, a, a, name="iou")
    check(abs(got - 1.0) < 1e-6,
          f"a box against itself gave {got:.4f}, expected 1.0.")
    far = (100.0, 100.0, 120.0, 120.0)
    check(
        abs(fn(a, far)) < 1e-6,
        f"two disjoint boxes gave {fn(a, far):.4f}, expected 0.0 -- a negative overlap "
        "must be clamped to zero before it is multiplied.",
    )
    b = (20.0, 10.0, 40.0, 30.0)   # half overlap: inter 200, union 600
    check(
        abs(fn(a, b) - 1 / 3) < 1e-6,
        f"two boxes overlapping in exactly half their area gave {fn(a, b):.4f}, "
        "expected 1/3 = 0.3333 -- the union is areaA + areaB - intersection, and "
        "subtracting the intersection is the step most often missed.",
    )

    from torchvision.ops import box_iou

    g = torch.Generator().manual_seed(0)
    xy = torch.rand(200, 2, generator=g) * 100
    wh = torch.rand(200, 2, generator=g) * 60 + 5
    boxes_a = torch.cat([xy, xy + wh], dim=1)
    xy2 = torch.rand(200, 2, generator=g) * 100
    wh2 = torch.rand(200, 2, generator=g) * 60 + 5
    boxes_b = torch.cat([xy2, xy2 + wh2], dim=1)
    worst = 0.0
    for i in range(200):
        ref = float(box_iou(boxes_a[i:i + 1], boxes_b[i:i + 1])[0, 0])
        worst = max(worst, abs(fn(tuple(boxes_a[i].tolist()), tuple(boxes_b[i].tolist())) - ref))
    check(
        worst < 1e-6,
        f"over 200 random pairs the largest disagreement with torchvision.ops.box_iou "
        f"was {worst:.2e}, expected below 1e-6.",
    )
    print("[ok] 2a: iou matches torchvision.ops.box_iou over 200 random pairs.")


def check_match(fn, gts, preds):
    """Exercise 2b."""
    pairs = _call(fn, gts[0], preds[0], name="match")
    check(
        all(len(p) == 3 for p in pairs),
        "match should return (gt_index, pred_index, iou) triples.",
    )

    total, used_total = 0, 0
    for g, p in zip(gts, preds):
        got = fn(g, p)
        idx = [pi for _, pi, _ in got]
        check(
            len(set(idx)) == len(idx),
            f"a prediction was matched to more than one ground-truth box on an image with "
            f"{len(g)} objects -- once a prediction is taken it must be struck off, or two "
            "overlapping objects will both claim it and recall will be overstated.",
        )
        for gi, pi, v in got:
            check(0 <= gi < len(g) and 0 <= pi < len(p),
                  f"match returned an index out of range: gt {gi}, pred {pi}.")
            check(v >= 0.5 - 1e-9,
                  f"match returned a pair at IoU {v:.3f}, below the 0.5 threshold.")
        total += len(got)
        used_total += len(idx)
    check(
        total == N_MATCHED,
        f"{total} of the {N_GT} ground-truth boxes matched, expected {N_MATCHED} -- "
        "each ground-truth box takes the UNUSED prediction of highest IoU, and the pair "
        "is kept only when that IoU reaches 0.5.",
    )
    print(f"[ok] 2b: {total} of {N_GT} ground-truth boxes matched, no prediction reused.")


def check_figure(fig, rows=4):
    """Exercise 2b, the figure."""
    check(fig is not None, "no figure was returned -- call show_comparison and keep its result.")
    axes = fig.get_axes()
    check(
        len(axes) == rows * 2,
        f"the figure has {len(axes)} panels, expected {rows * 2}: {rows} rows of "
        "ground truth beside predictions.",
    )
    empty = [i for i, ax in enumerate(axes) if not ax.patches]
    check(
        not empty,
        f"{len(empty)} of the {len(axes)} panels carry no boxes -- panels {empty[:4]} "
        "are showing the image alone.",
    )
    print(f"[ok] 2b: {rows} rows, both panels drawn, boxes on every panel.")


def check_metrics(recall, mean_iou, label_acc):
    """Exercise 2c."""
    check(0.35 <= recall <= 0.60,
          f"recall came out at {recall:.3f}, expected between 0.35 and 0.60 (measured "
          f"0.464) -- recall is matched / {N_GT} ground-truth boxes, not matched / "
          "predictions.")
    check(0.70 <= mean_iou <= 0.90,
          f"mean IoU came out at {mean_iou:.3f}, expected between 0.70 and 0.90 (measured "
          "0.826) -- average over MATCHED pairs only; unmatched boxes have no IoU to average.")
    check(label_acc >= 0.90,
          f"label accuracy came out at {label_acc:.3f}, expected above 0.90 (measured "
          "0.979) -- compare the predicted class of each matched pair with the ground "
          "truth's class.")
    print(f"[ok] 2c: recall {recall:.1%}, mean IoU {mean_iou:.3f}, "
          f"label accuracy {label_acc:.1%}.")
    print()
    print("     Read those three together. The detector names what it finds almost")
    print(f"     perfectly ({label_acc:.0%}), draws the box loosely (IoU {mean_iou:.2f}, not 1.0),")
    print(f"     and misses more than half the objects outright ({1 - recall:.0%}).")
    print("     Localisation and recall are the problem; classification is not.")
    print("     That is the error profile the unit gives for a single-shot detector.")


def check_sweep(rows):
    """Exercise 3a."""
    check(len(rows) == 4, f"the sweep has {len(rows)} rows, expected 4 thresholds.")
    confs = [r[0] for r in rows]
    check(confs == sorted(confs),
          f"the rows are at {confs}; keep them in increasing order of threshold.")
    recalls = [r[2] for r in rows]
    ious = [r[3] for r in rows]
    check(
        all(x >= y - 1e-9 for x, y in zip(recalls, recalls[1:])),
        f"recall went {[round(r, 3) for r in recalls]} as the threshold rose -- it should "
        "fall: a higher bar admits fewer boxes, so fewer objects are found.",
    )
    check(
        ious[-1] >= ious[0],
        f"mean IoU went {ious[0]:.3f} to {ious[-1]:.3f} as the threshold rose -- it should "
        "rise: the boxes that survive a high bar are the well-localised ones.",
    )
    check(
        recalls[0] >= 0.55,
        f"recall at the lowest threshold is {recalls[0]:.3f}, expected above 0.55 "
        "(measured 0.656).",
    )
    print(f"[ok] 3a: recall {recalls[0]:.1%} -> {recalls[-1]:.1%} and mean IoU "
          f"{ious[0]:.3f} -> {ious[-1]:.3f} as the threshold rises.")
    print("     The threshold buys precision with recall; it cannot buy both.")


def check_sweep_figure(fig):
    """Exercise 3a, the figure."""
    check(fig is not None, "no figure was returned -- keep the result of plt.gcf() or the fig.")
    lines = sum(len(ax.get_lines()) for ax in fig.get_axes())
    check(
        lines >= 2,
        f"the figure carries {lines} line(s), expected at least 2: recall and mean IoU "
        "against the threshold.",
    )
    print("[ok] 3a: both curves plotted.")


def check_scale(small_recall, large_recall, n_small, n_large):
    """Exercise 3b."""
    check(
        n_small + n_large == N_GT,
        f"the two groups hold {n_small + n_large} boxes, expected {N_GT} -- every "
        "ground-truth box belongs to exactly one group.",
    )
    check(
        n_small > n_large,
        f"{n_small} small and {n_large} large -- most COCO objects are small; a split "
        f"the other way suggests the area fraction is inverted (small is below "
        f"{SMALL_AREA:.0%} of the image).",
    )
    check(
        large_recall >= 2 * small_recall,
        f"large-object recall {large_recall:.3f} against small-object recall "
        f"{small_recall:.3f} -- expected the large group to be at least twice the small "
        "(measured 0.824 against 0.291). Is the area computed as a fraction of the "
        "IMAGE area rather than in pixels?",
    )
    print(f"[ok] 3b: recall {small_recall:.1%} on {n_small} small objects against "
          f"{large_recall:.1%} on {n_large} large ones, a factor of "
          f"{large_recall / max(small_recall, 1e-9):.1f}.")
    print()
    print("     The detector predicts from a coarse top-layer feature map, which carries")
    print("     little signal about a small object. This is the weakness SSD answered with")
    print("     prediction layers at several resolutions, and the reason the unit treats")
    print("     object scale as a design question rather than a detail.")
