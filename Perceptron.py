
import csv
import time
from collections import Counter

import numpy as np

import DataPreprocessing as dp
from DataPreprocessing import TIERS, prepare

# DataPreprocessing.py imports "numpy" but uses "np"
dp.np = np

SEED = 42

CAT_COLUMNS = [
    "category", "componentCategories", "region", "province",
    "contractor", "infraYear", "programName", "sourceOfFunds"
]


# Build one-hot features using training data only
# Budget is NOT a feature because the tier is computed from it
def build_features(train_projects, test_projects, top_k=100, min_count=20):
    vocab = []

    for col in CAT_COLUMNS:
        counts = Counter(p[col] for p in train_projects)

        for value, c in counts.most_common(top_k):
            if c >= min_count:
                vocab.append((col, value))

    index = {cv: i for i, cv in enumerate(vocab)}

    def encode(projects):
        X = np.zeros((len(projects), len(vocab)), dtype=np.float32)

        for r, p in enumerate(projects):
            for col in CAT_COLUMNS:
                j = index.get((col, p[col]))

                if j is not None:
                    X[r, j] = 1.0

        return X

    return encode(train_projects), encode(test_projects), vocab


class Perceptron:
    def __init__(self, eta=0.1, max_epochs=100, batch_size=1024,
                 pocket=True, patience=10, seed=SEED, verbose=False):

        self.eta = eta
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.pocket = pocket
        self.patience = patience
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

    # Standardize features and append bias column
    def _prep(self, X, fit=False):
        X = np.asarray(X, dtype=np.float32)

        if fit:
            self.mu = X.mean(axis=0)
            self.sd = X.std(axis=0)
            self.sd[self.sd == 0] = 1.0

        X = (X - self.mu) / self.sd

        return np.hstack([
            X,
            np.ones((len(X), 1), dtype=np.float32)
        ])

    # Multiclass averaged perceptron with decaying learning rate
    def fit(self, X, y_labels):
        X = self._prep(X, fit=True)

        n, m = X.shape
        k = len(TIERS)

        y = (
            np.asarray(y_labels)[:, None] == np.array(TIERS)[None, :]
        ).argmax(axis=1)

        eye = np.eye(k, dtype=np.float32)
        T = eye[y]

        W = np.zeros((k, m), dtype=np.float32)
        W_sum = np.zeros_like(W)

        steps = 0
        best_W = W.copy()
        best_err = np.inf
        stale = 0

        for epoch in range(self.max_epochs):

            lr = self.eta / (1 + epoch)
            order = self.rng.permutation(n)

            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]

                Xb = X[idx]

                P = eye[np.argmax(Xb @ W.T, axis=1)]

                W += lr * (T[idx] - P).T @ Xb / len(idx)

                W_sum += W
                steps += 1

            W_avg = W_sum / steps

            err = int(
                np.sum(np.argmax(X @ W_avg.T, axis=1) != y)
            )

            if err < best_err:
                best_err = err
                best_W = W_avg.copy()
                stale = 0
            else:
                stale += 1

            if self.verbose:
                print(
                    f"  epoch {epoch + 1:3d} "
                    f"errors={err} "
                    f"acc={1 - err / n:.4f} "
                    f"stale={stale}",
                    flush=True
                )

            if err == 0 or stale >= self.patience:
                break

        self.W = best_W if self.pocket else W_avg

        return self

    # Predict class with highest score
    def predict(self, X):
        X = self._prep(X)

        return np.array(TIERS)[
            np.argmax(X @ self.W.T, axis=1)
        ]


# Model evaluation
def evaluate(y_true, y_pred):

    idx = {t: i for i, t in enumerate(TIERS)}

    correct = 0
    total = len(y_true)

    tp = np.zeros(3)
    fp = np.zeros(3)
    fn = np.zeros(3)

    off_by_two = 0

    for actual, predicted in zip(y_true, y_pred):

        a = idx[actual]
        p = idx[predicted]

        if actual == predicted:
            correct += 1

        tp[p] += (a == p)
        fp[p] += (a != p)
        fn[a] += (a != p)

        if abs(a - p) == 2:
            off_by_two += 1

    accuracy = correct / total

    f1_scores = []

    for k in range(3):

        precision = tp[k] / (tp[k] + fp[k]) if tp[k] + fp[k] else 0
        recall = tp[k] / (tp[k] + fn[k]) if tp[k] + fn[k] else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        f1_scores.append(f1)

    macro_f1 = float(np.mean(f1_scores))
    off_by_two_rate = off_by_two / total

    return accuracy, macro_f1, off_by_two_rate


if __name__ == "__main__":

    # Prepare dataset
    data = prepare(seed=SEED)

    train_projects = data["train_projects"]
    test_projects = data["test_projects"]

    print(f"Kept {len(data['projects'])} rows")

    print(
        "Tier distribution:",
        dict(Counter(data["y_train"]) + Counter(data["y_test"]))
    )

    # Build features
    X_train, X_test, vocab = build_features(
        train_projects,
        test_projects
    )

    print(
        f"Train shape: {X_train.shape} | "
        f"Test shape: {X_test.shape}",
        flush=True
    )

    # Train model
    start = time.time()

    model = Perceptron(
        eta=0.1,
        max_epochs=100,
        verbose=True
    ).fit(X_train, data["y_train"])

    print(f"Training time: {time.time() - start:.2f}s")

    # Test predictions
    pred = model.predict(X_test)

    # Calculate metrics
    acc, mf1, o2 = evaluate(data["y_test"], pred)

    print(f"Accuracy      : {acc:.4f}")
    print(f"Macro-F1      : {mf1:.4f}")
    print(f"Off-by-two    : {o2:.4f} (Low<->High errors)")

    # Export predictions
    with open(
        "perceptron_predictions.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        w = csv.writer(f)

        w.writerow([
            "componentCategories",
            "region",
            "infraYear",
            "budget",
            "Actual Tier",
            "Predicted Tier",
            "Correct"
        ])

        for r, a, p in zip(test_projects, data["y_test"], pred):

            w.writerow([
                r["componentCategories"],
                r["region"],
                r["infraYear"],
                r["budget"],
                a,
                p,
                a == p\
            ])

    print("\nSaved perceptron_predictions.csv")
