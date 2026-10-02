"""Where the frozen probe sends source outputs it misclassifies: the target site (NYU) or another site.

Reads the canonical frozen-probe run's per-subject predictions. A low site accuracy reads as
harmonization only if errors point at the target; errors piled on one non-target class do not.
Protocol amendment 10 (post hoc description of existing predictions; nothing retrained).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

TARGET_SITE = 5  # NYU


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-run", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run = Path(args.eval_run)
    names = {}
    report = {}
    for path in sorted(run.glob("*_subjects.csv")):
        frame = pd.read_csv(path)
        names.update(dict(zip(frame.site_id, frame.subject_id.str.rsplit("_", n=1).str[0])))
        source = frame[frame.site_id != TARGET_SITE]
        wrong = source[source.harmonized_prediction != source.site_id]
        top = wrong.harmonized_prediction.value_counts()
        report[path.name.removesuffix("_subjects.csv")] = {
            "n_source": int(len(source)), "n_errors": int(len(wrong)),
            "errors_named_target": int((wrong.harmonized_prediction == TARGET_SITE).sum()),
            "target_share_of_errors": round(float((wrong.harmonized_prediction == TARGET_SITE).mean()), 3)
            if len(wrong) else None,
            "most_common_error": {"site": names.get(int(top.index[0])), "n": int(top.iloc[0])} if len(top) else None,
        }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    for method, r in report.items():
        print(f"{method:22s} errors {r['n_errors']:3d}/{r['n_source']}  named NYU {r['errors_named_target']:3d} "
              f"({r['target_share_of_errors']})  most common {r['most_common_error']}")


if __name__ == "__main__":
    main()
