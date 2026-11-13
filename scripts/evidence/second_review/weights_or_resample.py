"""7.4: apply A3's sampling weights exactly (a weighted confusion matrix) or by redrawing the
labelled set at random, or copying each label in proportion to its weight (so QuaPy can be used
unchanged)? Same simulation as
share_estimators.py: three senses, 1,000 usages per cell, labels stratified by predicted sense.
The agenda's numbers come from `python weights_or_resample.py 0.60,0.35,0.05` (a rare sense at 5%)."""

import numpy as np
from scipy.optimize import nnls

rng = np.random.default_rng(23)
CELL = 1000
import sys
SHARES = np.array([float(x) for x in sys.argv[1].split(",")]) if len(sys.argv) > 1 else np.array([0.40, 0.30, 0.30])
CONFUSION = np.array([[0.85, 0.10, 0.05], [0.10, 0.80, 0.05], [0.05, 0.10, 0.90]])  # P(pred | true)


def draw_cell() -> tuple[np.ndarray, np.ndarray]:
    truth = rng.choice(3, size=CELL, p=SHARES)
    cumulative = CONFUSION.cumsum(0)[:, truth]
    draw = rng.random(CELL)
    predicted = (draw > cumulative[0]).astype(int) + (draw > cumulative[1]).astype(int)
    return truth, predicted


def stratified_labels(predicted: np.ndarray, budget: int) -> tuple[np.ndarray, np.ndarray]:
    chosen, weights = [], []
    for sense in range(3):
        stratum = np.flatnonzero(predicted == sense)
        take = min(len(stratum), budget // 3)
        chosen.extend(rng.choice(stratum, size=take, replace=False))
        weights.extend([len(stratum) / take] * take)
    return np.array(chosen), np.array(weights)


def acc(predicted: np.ndarray, truth_l: np.ndarray, predicted_l: np.ndarray, weights: np.ndarray) -> float:
    confusion = np.zeros((3, 3))
    np.add.at(confusion, (predicted_l, truth_l), weights)
    totals = confusion.sum(0)
    confusion = np.where(totals > 0, confusion / np.where(totals > 0, totals, 1), np.eye(3))
    solution, _ = nnls(confusion, np.bincount(predicted, minlength=3) / len(predicted))
    return float(solution[2] / solution.sum())


def replicated(predicted: np.ndarray, truth_l: np.ndarray, predicted_l: np.ndarray,
               weights: np.ndarray, scale: int = 10) -> float:
    """Each label copied round(scale * weight / smallest weight) times, then unweighted ACC."""
    copies = np.maximum(1, np.rint(scale * weights / weights.min())).astype(int)
    return acc(predicted, np.repeat(truth_l, copies), np.repeat(predicted_l, copies),
               np.ones(copies.sum()))


def resampled(predicted: np.ndarray, truth_l: np.ndarray, predicted_l: np.ndarray,
              weights: np.ndarray, redraws: int) -> float:
    estimates = []
    for _ in range(redraws):
        index = rng.choice(len(truth_l), size=len(truth_l), p=weights / weights.sum())
        estimates.append(acc(predicted, truth_l[index], predicted_l[index], np.ones(len(index))))
    return float(np.mean(estimates))


for budget in (30, 60):
    errors = {"weighted (exact)": [], "replicated copies": [], "one redraw": [], "mean of 50 redraws": [],
              "weights ignored": []}
    for _ in range(2000):
        truth, predicted = draw_cell()
        index, weights = stratified_labels(predicted, budget)
        truth_l, predicted_l = truth[index], predicted[index]
        true_share = (truth == 2).mean()
        errors["weighted (exact)"].append(acc(predicted, truth_l, predicted_l, weights) - true_share)
        errors["replicated copies"].append(
            replicated(predicted, truth_l, predicted_l, weights) - true_share)
        errors["one redraw"].append(resampled(predicted, truth_l, predicted_l, weights, 1) - true_share)
        errors["mean of 50 redraws"].append(
            resampled(predicted, truth_l, predicted_l, weights, 50) - true_share)
        errors["weights ignored"].append(
            acc(predicted, truth_l, predicted_l, np.ones(len(index))) - true_share)
    print(f"{budget} labels per cell, shares {SHARES.tolist()}, error in the third sense's share:")
    for name, values in errors.items():
        values = np.array(values)
        print(f"  {name:<20} bias {values.mean():+.4f}   RMSE {np.sqrt((values ** 2).mean()):.4f}")
