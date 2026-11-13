"""A3 draws the author's labels at random from the cells themselves, stratified by period and by a
counter's predicted sense. With such labels there are two ways to estimate a sense's share in a
labelled cell:

- adjusted counting (7.4): estimate the counter's error rates from the labels, then correct its
  counts; assumes a sense is recognised equally well in every period unless rates are per period;
- the stratified estimator from survey sampling: within each predicted-sense stratum, the labelled
  share of each true sense, weighted by the stratum's size; assumes nothing about the counter.

Simulated: three senses, 1,000 usages per cell, a counter that is either equally good in both
periods or worse at the new sense in the early period (era bias)."""

import numpy as np
from scipy.optimize import nnls

rng = np.random.default_rng(11)
CELL = 1000
SHARES = {"early": np.array([0.60, 0.35, 0.05]), "late": np.array([0.40, 0.30, 0.30])}
GOOD = np.array([[0.85, 0.10, 0.05], [0.10, 0.80, 0.05], [0.05, 0.10, 0.90]])  # P(predicted | true), columns true
BIASED_EARLY = np.array([[0.85, 0.10, 0.30], [0.10, 0.80, 0.25], [0.05, 0.10, 0.45]])


def cell(shares: np.ndarray, confusion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = rng.choice(3, size=CELL, p=shares)
    cumulative = confusion.cumsum(0)[:, truth]
    draw = rng.random(CELL)
    predicted = (draw > cumulative[0]).astype(int) + (draw > cumulative[1]).astype(int)
    return truth, predicted


def stratified_labels(predicted: np.ndarray, budget: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal allocation over predicted senses; returns the labelled indices and their weights."""
    chosen, weights = [], []
    for sense in range(3):
        stratum = np.flatnonzero(predicted == sense)
        take = min(len(stratum), budget // 3)
        if take:
            chosen.extend(rng.choice(stratum, size=take, replace=False))
            weights.extend([len(stratum) / take] * take)
    return np.array(chosen), np.array(weights)


def adjusted_count(predicted: np.ndarray, truth_l: np.ndarray, predicted_l: np.ndarray, weights: np.ndarray) -> np.ndarray:
    confusion = np.zeros((3, 3))
    np.add.at(confusion, (predicted_l, truth_l), weights)
    column_totals = confusion.sum(0)
    confusion = np.where(column_totals > 0, confusion / np.where(column_totals > 0, column_totals, 1), np.eye(3))
    observed = np.bincount(predicted, minlength=3) / len(predicted)
    solution, _ = nnls(confusion, observed)
    return solution / solution.sum() if solution.sum() > 0 else observed


def stratified(predicted: np.ndarray, truth_l: np.ndarray, predicted_l: np.ndarray) -> np.ndarray:
    estimate = np.zeros(3)
    for sense in range(3):
        size = (predicted == sense).mean()
        labelled = truth_l[predicted_l == sense]
        if len(labelled):
            estimate += size * np.bincount(labelled, minlength=3) / len(labelled)
        else:
            estimate[sense] += size
    return estimate


def run(era_bias: bool, budget: int, repeats: int = 3000) -> None:
    errors: dict[str, list[float]] = {name: [] for name in (
        "classify and count", "adjusted, pooled, weights ignored", "adjusted, pooled error rates", "adjusted, per-period error rates", "stratified estimator")}
    for _ in range(repeats):
        cells = {
            "early": cell(SHARES["early"], BIASED_EARLY if era_bias else GOOD),
            "late": cell(SHARES["late"], GOOD),
        }
        labels = {period: stratified_labels(predicted, budget) for period, (_, predicted) in cells.items()}
        pooled_truth = np.concatenate([cells[p][0][labels[p][0]] for p in cells])
        pooled_predicted = np.concatenate([cells[p][1][labels[p][0]] for p in cells])
        pooled_weights = np.concatenate([labels[p][1] for p in cells])
        estimates: dict[str, dict[str, float]] = {name: {} for name in errors}
        for period, (truth, predicted) in cells.items():
            index, weights = labels[period]
            estimates["classify and count"][period] = (predicted == 2).mean()
            estimates["adjusted, pooled, weights ignored"][period] = adjusted_count(
                predicted, pooled_truth, pooled_predicted, np.ones(len(pooled_weights)))[2]
            estimates["adjusted, pooled error rates"][period] = adjusted_count(
                predicted, pooled_truth, pooled_predicted, pooled_weights)[2]
            estimates["adjusted, per-period error rates"][period] = adjusted_count(
                predicted, truth[index], predicted[index], weights)[2]
            estimates["stratified estimator"][period] = stratified(predicted, truth[index], predicted[index])[2]
        true_change = (cells["late"][0] == 2).mean() - (cells["early"][0] == 2).mean()
        for name in errors:
            errors[name].append(estimates[name]["late"] - estimates[name]["early"] - true_change)
    print(f"  {'counter worse at the new sense early' if era_bias else 'counter equally good in both periods':<38}"
          f" {budget} labels per cell")
    for name, values in errors.items():
        values = np.array(values)
        print(f"      {name:<34} bias {values.mean():+.3f}   RMSE {np.sqrt((values**2).mean()):.3f}")


print("Error in the estimated rise of the new sense's share (true rise about 0.25):")
for era_bias in (False,):
    for budget in (30, 60):
        run(era_bias, budget)
