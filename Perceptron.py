import csv
import time

import numpy as np

from DataPreprocessing import SEED, TIERS, prepare


# Sign function
def sgn(v):
    return np.where(v > 0, 1, -1)


class Perceptron:
    def __init__(self, eta=0.1, max_epochs=100, batch_size=256,
                 pocket=True, patience=10, seed=SEED):
        self.eta = eta
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.pocket = pocket
        self.patience = patience
        self.rng = np.random.default_rng(seed)

    # Model training (all three tier neurons are updated together)
    def fit(self, X, y_labels):
        n, m = X.shape
        D = np.where(y_labels[:, None] == np.array(TIERS)[None, :], 1, -1)

        W = np.zeros((len(TIERS), m))
        best_W = W.copy()
        best_err = np.full(len(TIERS), np.inf)
        stale = 0

        for _ in range(self.max_epochs):
            order = self.rng.permutation(n)

            # Error-correction rule applied on mini-batches
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                Y = sgn(X[idx] @ W.T)
                W += self.eta * (D[idx] - Y).T @ X[idx] / len(idx)

            # Pocket selection per neuron
            err = np.sum(sgn(X @ W.T) != D, axis=0)
            improved = err < best_err
            if improved.any():
                best_W[improved] = W[improved]
                best_err[improved] = err[improved]
                stale = 0
            else:
                stale += 1

            # Stop when separable or no longer improving
            if err.sum() == 0 or stale >= self.patience:
                break

        self.W = best_W if self.pocket else W
        return self

    # Class prediction
    def predict(self, X):
        v = X @ self.W.T
        return np.array(TIERS)[np.argmax(v, axis=1)]


# Model evaluation
def evaluate(y_true, y_pred):
    idx = {t: i for i, t in enumerate(TIERS)}
    cm = np.zeros((3, 3), dtype=int)

    for a, p in zip(y_true, y_pred):
        cm[idx[a], idx[p]] += 1

    acc = np.trace(cm) / cm.sum()
    f1s = []

    for k in range(3):
        tp = cm[k, k]
        prec = tp / cm[:, k].sum() if cm[:, k].sum() else 0
        rec = tp / cm[k, :].sum() if cm[k, :].sum() else 0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0)

    off_by_two = (cm[0, 2] + cm[2, 0]) / cm.sum()

    return acc, float(np.mean(f1s)), off_by_two


if __name__ == "__main__":
    # Prepare dataset
    data = prepare()

    # Train model
    start = time.time()
    model = Perceptron(eta=0.1, max_epochs=100).fit(
        data["X_train"], data["y_train"]
    )
    print(f"Training time : {time.time() - start:.2f}s")

    # Test predictions
    pred = model.predict(data["X_test"])

    # Calculate metrics
    acc, mf1, o2 = evaluate(data["y_test"], pred)

    print(f"Accuracy      : {acc:.4f}")
    print(f"Macro-F1      : {mf1:.4f}")
    print(f"Off-by-two    : {o2:.4f}  (Low<->High errors)")

    # Export predictions
    with open("perceptron_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["componentCategories", "region", "infraYear", "budget",
                    "Actual Tier", "Predicted Tier", "Correct"])

        for r, a, p in zip(data["test_records"], data["y_test"], pred):
            w.writerow([r["type"], r["region"], r["year"], r["budget"], a, p, a == p])

    print("\nSaved perceptron_predictions.csv")