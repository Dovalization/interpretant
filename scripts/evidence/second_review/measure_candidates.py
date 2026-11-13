"""J is the two-sample statistic with a linear kernel. Do the standard two-sample statistics that
see more than the mean do better on DWUG EN, while still falling to zero when periods are shuffled?
Development data only (DWUG EN is experiment 0a's material)."""

from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

from interpretant.drift.pairs import dwug_graded_change

ROOT = Path("data/external/dwug_en_2.0.1/dwug_en")
stored = np.load("data/drift/pairs_dwug_en_2.0.1.npz")
gold = dwug_graded_change(ROOT)
words = sorted({key.split("|")[0] for key in stored.files})


def unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def off_diagonal_mean(matrix: np.ndarray) -> float:
    count = len(matrix)
    return float((matrix.sum() - np.trace(matrix)) / (count * (count - 1)))


def measures(first: np.ndarray, second: np.ndarray, with_classifier: bool) -> dict[str, float]:
    a, b = unit(first), unit(second)
    cosine_aa, cosine_bb, cosine_ab = 1 - a @ a.T, 1 - b @ b.T, 1 - a @ b.T
    apd = float(cosine_ab.mean())
    within = (off_diagonal_mean(cosine_aa) + off_diagonal_mean(cosine_bb)) / 2
    euclid = lambda matrix: np.sqrt(np.clip(2 * matrix, 0, None))  # noqa: E731
    energy = 2 * euclid(cosine_ab).mean() - off_diagonal_mean(euclid(cosine_aa)) - off_diagonal_mean(
        euclid(cosine_bb)
    )
    bandwidth = np.median(np.concatenate([cosine_aa.ravel(), cosine_bb.ravel(), cosine_ab.ravel()]))
    kernel = lambda matrix: np.exp(-matrix / bandwidth)  # noqa: E731
    rbf = (
        off_diagonal_mean(kernel(cosine_aa))
        + off_diagonal_mean(kernel(cosine_bb))
        - 2 * kernel(cosine_ab).mean()
    )
    mean_a, mean_b = a.mean(0), b.mean(0)
    result = {
        "APD": apd,
        "PRT": float(1 - mean_a @ mean_b / np.linalg.norm(mean_a) / np.linalg.norm(mean_b)),
        "J (linear kernel)": apd - within,
        "J / within (ratio)": (apd - within) / within,
        "energy distance": float(energy),
        "MMD, Gaussian kernel": float(rbf),
    }
    if with_classifier:
        data = np.vstack([a, b])
        labels = np.r_[np.zeros(len(a)), np.ones(len(b))]
        scores = cross_val_predict(
            LogisticRegression(C=1.0, max_iter=2000), data, labels, cv=5, method="decision_function"
        )
        result["period classifier AUC"] = float(roc_auc_score(labels, scores))
    return result


real = {word: measures(stored[f"{word}|1"], stored[f"{word}|2"], True) for word in words}
names = list(next(iter(real.values())))
target = [gold[word] for word in words]

rng = np.random.default_rng(2026)
shuffled_agreement = {name: [] for name in names if name != "period classifier AUC"}
for _ in range(20):
    run = {}
    for word in words:
        pooled = np.vstack([stored[f"{word}|1"], stored[f"{word}|2"]])
        order = rng.permutation(len(pooled))
        cut = len(stored[f"{word}|1"])
        run[word] = measures(pooled[order[:cut]], pooled[order[cut:]], False)
    for name in shuffled_agreement:
        shuffled_agreement[name].append(
            stats.spearmanr(target, [run[word][name] for word in words]).statistic
        )

print(f"{'measure':<26}{'agreement':>10}{'periods shuffled':>18}")
for name in names:
    agreement = stats.spearmanr(target, [real[word][name] for word in words]).statistic
    shuffled = shuffled_agreement.get(name)
    shown = f"{np.mean(shuffled):+.3f}" if shuffled else "not run"
    print(f"{name:<26}{agreement:>10.3f}{shown:>18}")

print("\nrank correlation between the measures (real periods):")
matrix = np.array([[real[word][name] for name in names] for word in words])
correlation = stats.spearmanr(matrix).statistic
print(" " * 26 + "".join(f"{index:>6}" for index in range(len(names))))
for index, name in enumerate(names):
    print(f"{index} {name:<24}" + "".join(f"{value:>6.2f}" for value in correlation[index]))
