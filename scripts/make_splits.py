import json
from pathlib import Path

import pandas as pd


def main():
    manifest_path = Path("data/abide_manifest.csv")
    out_path = Path("data/splits.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)

    # unique subjects with site
    subj = (
        df[["subject_id", "site"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # stratified split by site to avoid site distribution drift
    seed = 42
    train_frac, val_frac, test_frac = 0.8, 0.1, 0.1
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9

    train_ids, val_ids, test_ids = [], [], []

    for site, g in subj.groupby("site"):
        g = g.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle
        n = len(g)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        # ensure totals don't exceed n
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        n_test = n - n_train - n_val

        train_ids.extend(g.iloc[:n_train]["subject_id"].tolist())
        val_ids.extend(g.iloc[n_train:n_train + n_val]["subject_id"].tolist())
        test_ids.extend(g.iloc[n_train + n_val:]["subject_id"].tolist())

    split = {
        "seed": seed,
        "fractions": {"train": train_frac, "val": val_frac, "test": test_frac},
        "counts": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }

    out_path.write_text(json.dumps(split, indent=2))
    print(f"Wrote splits -> {out_path}")
    print(split["counts"])


if __name__ == "__main__":
    main()