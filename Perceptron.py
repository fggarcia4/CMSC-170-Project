import csv

import numpy as np

from DataPreprocessing import SEED, TIERS, prepare


# Sign function
def sgn(v):
    return np.where(v > 0, 1, -1)


class Perceptron:
    def __init__(self, eta=0.1, max_epochs=100, pocket=True, seed=SEED):
        self.eta, self.max_epochs, self.pocket = eta, max_epochs, pocket
        self.rng = np.random.default_rng(seed)

    # Binary training
    def _train_binary(self, X, d):
        w = np.zeros(X.shape[1])
        best_w, best_err = w.copy(), np.inf

        for _ in range(self.max_epochs):
            changed = False

            for i in self.rng.permutation(len(X)):
                y = 1 if X[i] @ w > 0 else -1

                if y != d[i]:
                    w = w + self.eta * (d[i] - y) * X[i]
                    changed = True

            # Pocket selection
            if self.pocket:
                err = np.sum(sgn(X @ w) != d)
                if err < best_err:
                    best_err, best_w = err, w.copy()

            if not changed:
                return w

        return best_w if self.pocket else w

    # Model training
    def fit(self, X, y_labels):
        self.W = np.vstack([
            self._train_binary(X, np.where(y_labels == t, 1, -1)) for t in TIERS
        ])
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
    model = Perceptron(eta=0.1, max_epochs=100).fit(
        data["X_train"], data["y_train"]
    )

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