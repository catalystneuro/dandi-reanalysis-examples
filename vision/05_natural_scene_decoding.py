"""Step 4, the picture: can we tell which photograph the mouse saw from its neurons?

118 natural photographs were flashed for 250 ms each, about 50 times each in
random order. For each unit we count spikes 30-280 ms after image onset. A Poisson
naive-Bayes decoder is trained on some repeats and asked to name the image on
held-out repeats (5-fold cross-validation over repeats). Chance is 1/118 = 0.85%.

Two views:
* Within session: 30 randomly drawn units per area (20 draws), so areas are
  compared with the same population size.
* Pseudo-population: because every mouse saw the same 118 images, repeat k of an
  image in one mouse can be paired with repeat k in another. This pools units
  across mice to ask how accuracy grows with the number of neurons.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import AREA_LABELS, AREAS, SESSIONS, counts_matrix, load_session
from plotting import AREA_COLORS, FIG_DIR, RESULTS_DIR, style

WINDOW = (0.03, 0.28)
N_REPEATS = 45  # repeats per image kept for every session (each has ~50)
N_UNITS_SESSION = 30
N_DRAWS = 20
N_FOLDS = 5
POP_SIZES = [1, 3, 10, 30, 100, 300]
N_CONFUSION = 100  # V1 units used for the confusion matrix


def image_tensor(units, stims, invalid):
    """Counts as (images, repeats, units), dropping units with any invalid trial in the block."""
    ns = stims["natural_scenes"]
    frame = ns.metadata["frame"].values
    counts = counts_matrix(units, invalid, ns.start, WINDOW)
    ok = ~np.isnan(counts).any(0)
    images = np.arange(118)
    X = np.stack([counts[frame == im][:N_REPEATS] for im in images])
    return X[:, :, ok], np.asarray(units.metadata["area"].values)[ok]


def poisson_nb_accuracy(X, rng):
    """Cross-validated accuracy of a Poisson naive-Bayes decoder; X is (images, repeats, units)."""
    n_img, n_rep, _ = X.shape
    fold = rng.permutation(n_rep) % N_FOLDS
    correct = 0
    for f in range(N_FOLDS):
        train, test = X[:, fold != f], X[:, fold == f]
        lam = train.mean(1) + 1e-3  # (images, units)
        # log-likelihood of each test trial under each image: sum_u k log(lam) - lam
        ll = np.einsum("itu,ju->itj", test, np.log(lam)) - lam.sum(1)
        pred = ll.argmax(2)
        correct += np.sum(pred == np.arange(n_img)[:, None])
    return correct / (n_img * n_rep)


def main():
    rng = np.random.default_rng(0)
    rows, pools = [], {a: [] for a in AREAS}
    for sid in tqdm(SESSIONS, desc="natural scenes"):
        units, stims, invalid = load_session(sid)
        X, area = image_tensor(units, stims, invalid)
        for a in AREAS:
            Xa = X[:, :, area == a]
            if Xa.shape[2] == 0:
                continue
            pools[a].append(Xa)
            if Xa.shape[2] < N_UNITS_SESSION:
                continue
            acc = [
                poisson_nb_accuracy(Xa[:, :, rng.choice(Xa.shape[2], N_UNITS_SESSION, replace=False)], rng)
                for _ in range(N_DRAWS)
            ]
            rows.append({"session": sid, "area": a, "n_units_available": Xa.shape[2], "accuracy": np.mean(acc)})
    per_session = pd.DataFrame(rows)
    per_session.to_csv(RESULTS_DIR / "decoding_per_session.csv", index=False)
    print(per_session.groupby("area")["accuracy"].agg(["count", "median", "min", "max"]).reindex(AREAS).round(3))

    curve_rows = []
    for a in AREAS:
        P = np.concatenate(pools[a], axis=2)
        for n in POP_SIZES:
            if n > P.shape[2]:
                continue
            draws = 10 if n < 300 else 3
            for _ in range(draws):
                acc = poisson_nb_accuracy(P[:, :, rng.choice(P.shape[2], n, replace=False)], rng)
                curve_rows.append({"area": a, "n_units": n, "accuracy": acc, "pool_size": P.shape[2]})
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(RESULTS_DIR / "decoding_vs_population_size.csv", index=False)
    print(curve.groupby(["area", "n_units"])["accuracy"].mean().unstack().reindex(AREAS).round(3))

    # Confusion matrix for a 100-unit V1 pseudo-population (about 80% correct,
    # so the errors are visible)
    P = np.concatenate(pools["VISp"], axis=2)
    P = P[:, :, rng.choice(P.shape[2], N_CONFUSION, replace=False)]
    n_img, n_rep, _ = P.shape
    fold = rng.permutation(n_rep) % N_FOLDS
    conf = np.zeros((n_img, n_img))
    for f in range(N_FOLDS):
        lam = P[:, fold != f].mean(1) + 1e-3
        ll = np.einsum("itu,ju->itj", P[:, fold == f], np.log(lam)) - lam.sum(1)
        pred = ll.argmax(2)
        for i in range(n_img):
            np.add.at(conf[i], pred[i], 1)
    conf /= conf.sum(1, keepdims=True)
    np.save(RESULTS_DIR / "decoding_confusion_v1.npy", conf)
    plot(per_session, curve, conf, P.shape[2])


def plot(per_session, curve, conf, n_v1):
    fig = plt.figure(figsize=(14, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1], wspace=0.35)
    chance = 1 / 118

    ax = fig.add_subplot(gs[0])
    for i, a in enumerate(AREAS):
        v = per_session.loc[per_session.area == a, "accuracy"].values * 100
        if len(v) == 0:
            continue
        x = i + (np.random.default_rng(1).random(len(v)) - 0.5) * 0.3
        ax.scatter(x, v, s=22, color=AREA_COLORS[a], edgecolor="white", lw=0.5, zorder=2)
        ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color="k", lw=2, zorder=3)
    ax.axhline(chance * 100, color="0.5", ls="--", lw=1)
    ax.text(len(AREAS) - 0.5, chance * 100 + 0.6, "chance", fontsize=8, color="0.4", ha="right")
    ax.set_xticks(range(len(AREAS)), [AREA_LABELS[a].split(" ")[0] for a in AREAS])
    ax.set_ylabel("images correctly named (%)")
    ax.set_title(f"{N_UNITS_SESSION} units per area, one mouse at a time\n(dots = mice, bar = median)", fontsize=10)
    style(ax)

    ax = fig.add_subplot(gs[1])
    for a in AREAS:
        c = curve[curve.area == a].groupby("n_units")["accuracy"].mean() * 100
        if c.empty:
            continue
        ax.plot(c.index, c.values, "-o", ms=4, color=AREA_COLORS[a], lw=2, label=AREA_LABELS[a].split(" ")[0])
    ax.axhline(chance * 100, color="0.5", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xticks(POP_SIZES, [str(n) for n in POP_SIZES])
    ax.minorticks_off()
    ax.set_xlabel("number of units (pooled across mice)")
    ax.set_ylabel("images correctly named (%)")
    ax.set_title("More neurons, sharper picture", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    style(ax)

    ax = fig.add_subplot(gs[2])
    im = ax.imshow(conf * 100, cmap="magma", vmin=0, vmax=100)
    ax.set_xlabel("decoded image")
    ax.set_ylabel("shown image")
    ax.set_title(f"V1 confusion matrix, {n_v1} units\n(diagonal = correct)", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("% of presentations", fontsize=8)
    fig.savefig(FIG_DIR / "05_natural_scene_decoding.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
