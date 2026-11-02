import ast
import csv
import math
import os
import random
import sys

import numpy as np

TIERS = ["Low", "Middle", "High"]
CSV_NAME = "dpwh_transparency_data.csv"
SEED = 42
LOW_MAX = 5_000_000
MIDDLE_MAX = 8_999_999


def find_csv(argv=None):
    argv = sys.argv if argv is None else argv
    if len(argv) > 1:
        return argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.path.join(here, CSV_NAME), CSV_NAME):
        if os.path.exists(candidate):
            return candidate
    sys.exit(f"Cannot find {CSV_NAME}. Put it in the same folder as the scripts.")


def budget_to_tier(budget):
    if budget <= LOW_MAX:
        return "Low"
    if budget <= MIDDLE_MAX:
        return "Middle"
    return "High"


def parse_region(location):
    try:
        region = ast.literal_eval(location)["region"]
        region = str(region).strip()
        return region if region else None
    except Exception:
        return None


def parse_number(value):
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clean_records(rows):
    records = []
    dropped = {"missing_type": 0, "missing_region": 0, "bad_year": 0, "bad_budget": 0}
    for row in rows:
        ptype = (row.get("componentCategories") or "").strip()
        if not ptype:
            dropped["missing_type"] += 1
            continue
        region = parse_region(row.get("location"))
        if region is None:
            dropped["missing_region"] += 1
            continue
        year = parse_number(row.get("infraYear"))
        if year is None:
            dropped["bad_year"] += 1
            continue
        budget = parse_number(row.get("budget"))
        if budget is None or budget <= 0:
            dropped["bad_budget"] += 1
            continue
        records.append({
            "type": ptype,
            "region": region,
            "year": int(year),
            "budget": budget,
            "tier": budget_to_tier(budget),
        })
    return records, dropped


def load_dataset(path, verbose=True):
    rows = load_csv(path)
    records, dropped = clean_records(rows)
    if verbose:
        print(f"Loaded {len(rows)} rows from {path}")
        print(f"Kept {len(records)} rows after cleaning")
        for reason, count in dropped.items():
            print(f"  dropped ({reason}): {count}")
        counts = {t: 0 for t in TIERS}
        for r in records:
            counts[r["tier"]] += 1
        print("Tier distribution:", counts, "\n")
    return records


def get_labels(records):
    return np.array([r["tier"] for r in records])


def stratified_split(labels, test_frac=0.2, seed=SEED):
    rng = random.Random(seed)
    by_class = {}
    for i, label in enumerate(labels):
        by_class.setdefault(label, []).append(i)
    train_idx, test_idx = [], []
    for label in sorted(by_class):
        idx = by_class[label][:]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_frac))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx), np.array(test_idx)


def subset(records, idx):
    return [records[i] for i in idx]


class Encoder:
    def __init__(self, bias=True):
        self.bias = bias

    def fit(self, records):
        self.types = sorted({r["type"] for r in records})
        self.regions = sorted({r["region"] for r in records})
        years = [r["year"] for r in records]
        self.year_min = min(years)
        self.year_max = max(years)
        self.type_index = {t: i for i, t in enumerate(self.types)}
        self.region_index = {g: i for i, g in enumerate(self.regions)}
        return self

    @property
    def feature_names(self):
        names = ["bias"] if self.bias else []
        names += [f"type={t}" for t in self.types]
        names += [f"region={g}" for g in self.regions]
        names.append("year_scaled")
        return names

    def scale_year(self, year):
        span = self.year_max - self.year_min
        return (year - self.year_min) / span if span else 0.0

    def transform(self, records):
        offset = 1 if self.bias else 0
        n_types, n_regions = len(self.types), len(self.regions)
        X = np.zeros((len(records), offset + n_types + n_regions + 1))
        if self.bias:
            X[:, 0] = 1.0
        for row, r in enumerate(records):
            t = self.type_index.get(r["type"])
            g = self.region_index.get(r["region"])
            if t is not None:
                X[row, offset + t] = 1.0
            if g is not None:
                X[row, offset + n_types + g] = 1.0
            X[row, -1] = self.scale_year(r["year"])
        return X


def prepare(path=None, test_frac=0.2, seed=SEED, bias=True, verbose=True):
    path = path or find_csv()
    records = load_dataset(path, verbose)
    labels = get_labels(records)
    train_idx, test_idx = stratified_split(labels, test_frac, seed)
    train_records = subset(records, train_idx)
    test_records = subset(records, test_idx)
    encoder = Encoder(bias=bias).fit(train_records)
    return {
        "records": records,
        "train_records": train_records,
        "test_records": test_records,
        "y_train": labels[train_idx],
        "y_test": labels[test_idx],
        "X_train": encoder.transform(train_records),
        "X_test": encoder.transform(test_records),
        "encoder": encoder,
        "test_idx": test_idx,
    }
if __name__ == "__main__":
    data = prepare()
    print("Train:", data["X_train"].shape, "Test:", data["X_test"].shape)
    print("Features:", len(data["encoder"].feature_names))
    print("Sample feature vector:", data["X_train"][0])
    print("Sample label:", data["y_train"][0])