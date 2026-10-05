"""How much site is decodable from demographics and brain size alone (protocol amendment 11).

Multinomial logistic regression on standardized inputs with balanced class weights, trained on
the training subjects and scored on the source (non-NYU) test subjects with the evaluator's
site-stratified bootstrap. Inputs come from `analysis/demographics_inputs.csv`: ABIDE I age, sex
and diagnosis, and FastSurfer total brain volume computed from the raw T1 volumes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from harmonit.metrics.subject_evaluation import balanced_accuracy_draws, stratified_indices  # noqa: E402

TARGET_SITE = 5
FEATURES = {"age_sex_diagnosis": ["age", "sex", "dx"], "age_sex_diagnosis_brain_volume": ["age", "sex", "dx", "vol_brain"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    data = pd.read_csv(args.inputs)
    train = data[data.split == "train"]
    test = data[(data.split == "test") & (data.site_id != TARGET_SITE)].reset_index(drop=True)
    indices = stratified_indices(test.site_id.to_numpy())
    report = {"n_train": int(len(train)), "n_source_test": int(len(test)), "results": {}}
    for name, cols in FEATURES.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
        model.fit(train[cols], train.site_id)
        est, draws = balanced_accuracy_draws(test.site_id.to_numpy(), model.predict(test[cols]), indices)
        report["results"][name] = {"features": cols, "source_ba": round(float(est), 4),
                                   "ci95": [round(float(v), 4) for v in np.quantile(draws, [.025, .975])]}
        print(f"{name:32s} source BA {est:.3f} {report['results'][name]['ci95']}")
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
