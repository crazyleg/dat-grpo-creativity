"""Recompute the paired story scores from released final judgments; no API calls."""
from collections import defaultdict
import json
from pathlib import Path
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def interval(values, clusters):
    grouped = defaultdict(list)
    for value, cluster in zip(values, clusters):
        grouped[cluster].append(value)
    means = np.array([np.mean(group) for group in grouped.values()])
    rng = np.random.default_rng(20261103)
    draws = rng.integers(0, len(means), size=(5000, len(means)))
    low, high = np.quantile(means[draws].mean(axis=1), [.025, .975])
    return {"mean": float(means.mean()), "ci95_low": float(low), "ci95_high": float(high)}


def main():
    expected = json.loads((ROOT / "results/summary/sol-results.json").read_text())["metrics"]
    with tarfile.open(ROOT / "results/raw/sol-story-judgments.tar.gz") as bundle:
        jobs = json.load(bundle.extractfile("jobs.json"))
        pairs = defaultdict(dict)
        for job in jobs:
            name = job["case_id"].replace("/", "-").replace(":", "-order-")
            answer = json.load(bundle.extractfile(f"judgments/{name}/answer.json"))
            pairs[job["writer_case"]][job["order"]] = (job["cluster_id"], answer)
    if len(jobs) != 128 or len(pairs) != 64 or any(set(v) != {0, 1} for v in pairs.values()):
        raise ValueError("Expected 64 complete reversed-order pairs")
    result = {}
    for metric in expected:
        before, after, credits, clusters = [], [], [], []
        for orders in pairs.values():
            b, f, w = [], [], []
            for order in [0, 1]:
                cluster, answer = orders[order]
                a, z = answer["story_a"][metric], answer["story_b"][metric]
                b.append(a if order == 0 else z)
                f.append(z if order == 0 else a)
                winner = answer["winner"] if metric == "overall_quality" else "TIE" if abs(a-z) < .5 else "A" if a > z else "B"
                w.append(.5 if winner == "TIE" else float(winner == ("B" if order == 0 else "A")))
            before.append(float(np.mean(b)))
            after.append(float(np.mean(f)))
            credits.append(float(np.mean(w)))
            clusters.append(cluster)
        row = {"baseline_mean": float(np.mean(before)), "full250_mean": float(np.mean(after)),
               "delta": interval(np.array(after)-np.array(before), clusters), "win_credit": interval(credits, clusters)}
        for key in ["baseline_mean", "full250_mean"]:
            if not np.isclose(row[key], expected[metric][key], rtol=0, atol=1e-12):
                raise ValueError(f"Published mean differs for {metric}/{key}")
        for kind in ["delta", "win_credit"]:
            for key in ["mean", "ci95_low", "ci95_high"]:
                if not np.isclose(row[kind][key], expected[metric][kind][key], rtol=0, atol=1e-12):
                    raise ValueError(f"Published interval differs for {metric}/{kind}/{key}")
        result[metric] = row
    print(json.dumps(result, indent=2))
    print("All seven metrics reproduce the published means and clustered intervals.")


if __name__ == "__main__":
    main()
