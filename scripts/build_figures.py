"""Regenerate publication figures and Markdown tables from released results."""
import csv
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "article/figures"
INK = "#243240"
BASE = "#77848f"
LORA = "#b48439"
FULL = "#227c78"
LOSS = "#b25743"
BG = "#fbfaf7"


def rounded(value, places=2):
    return str(Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def load(name):
    return json.loads((ROOT / "results/summary" / name).read_text())


def save(fig, name, *, tight=True):
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=190, facecolor=BG,
                    bbox_inches="tight" if tight else None)
    plt.close(fig)


def poem_figure():
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.text(.5, .93, "Two poems, baseline QWEN vs creativity-tuned model",
             fontsize=23, weight="bold", color=INK, ha="center")
    for x, model, label in [(.055, "baseline", "Baseline QWEN"),
                             (.525, "full250", "Creativity-tuned model")]:
        fig.text(x, .825, label, fontsize=14, weight="bold", color=INK)
        raw = (ROOT / "results/samples" / f"sonnet-{model}.txt").read_text().strip()
        # Omit the model-generated heading; preserve all fourteen poem lines.
        if raw.startswith("**"):
            _, raw = raw.split("\n", 1)
            raw = raw.strip()
        lines = [line for line in raw.splitlines() if line.strip()]
        if len(lines) != 14:
            raise ValueError(f"Expected fourteen poem lines, found {len(lines)}")
        for i, line in enumerate(lines):
            stanza_gap = .012 * (i // 4)
            fig.text(x, .75 - i * .043 - stanza_gap, line, fontsize=13,
                     color=INK, family="DejaVu Serif")
    fig.add_artist(plt.Line2D([.5, .5], [.12, .85], color="#d9ddd8", linewidth=1))
    # A fixed canvas keeps the two columns equally wide despite line lengths.
    save(fig, "eternal-love-comparison", tight=False)


def judge_figure():
    metrics = load("sol-results.json")["metrics"]
    labels = {
        "creativity": "Creativity", "imagery": "Imagery", "style": "Style",
        "coherence": "Coherence", "character_and_impact": "Character and impact",
        "integration": "Element integration", "overall_quality": "Overall quality",
    }
    fig, ax = plt.subplots(figsize=(10.7, 6.6), facecolor=BG)
    ax.set_facecolor(BG)
    for i, (key, label) in enumerate(labels.items()):
        d = metrics[key]["delta"]
        color = FULL if d["ci95_low"] > 0 else LOSS if d["ci95_high"] < 0 else BASE
        ax.errorbar(d["mean"], i, xerr=[[d["mean"] - d["ci95_low"]], [d["ci95_high"] - d["mean"]]], fmt="o", color=color, capsize=5, markersize=8, linewidth=2.3)
        ax.text(1.08, i, f'{d["mean"]:+.2f}', va="center", color=color, fontsize=12, weight="bold")
    ax.axvline(0, color="#c3c9c8", linewidth=1)
    ax.set_yticks(range(len(labels)), labels.values(), fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(-1.5, 1.32)
    ax.set_xlabel("Full250 minus baseline, on a 0–10 score scale", fontsize=11, labelpad=15)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.grid(axis="x", color="#e8e8e2", alpha=.7)
    ax.set_axisbelow(True)
    fig.suptitle("Higher judged creativity, lower coherence", fontsize=20, weight="bold", color=INK, x=.07, ha="left", y=.99)
    fig.text(.07, .915, "64 matched story pairs  ·  both A/B orders  ·  gpt-6.1-sol, medium reasoning", fontsize=11, color=BASE)
    fig.text(.07, .015, "Unadjusted 95% intervals: 5,000 bootstrap resamples of 32 brief clusters. One judge and a fixed panel.", fontsize=9, color=BASE)
    fig.subplots_adjust(top=.83, left=.25, bottom=.16, right=.95)
    save(fig, "creative-transfer")


def dat_figure():
    full = load("dat-full-results.json")["scores"]["no_think"]
    lora = load("dat-lora-results.json")["scores"]["no_think"]
    counts_full = load("dat-full-class-counts.json")["models"]
    counts_lora = load("dat-lora-class-counts.json")["models"]
    colors = [BASE, LORA, FULL]
    labels = ["Baseline", "LoRA1500", "Full250"]
    x = np.arange(3)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 6.5), facecolor=BG)
    for ax in (a, b):
        ax.set_facecolor(BG)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="y", color="#e8e8e2")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    for j, (label, color, scores) in enumerate(zip(labels, colors, [full["original"], lora["step1500"], full["step250"]])):
        values = [scores[f"dat_{c}/reward_score"]["mean"] for c in ["nouns", "verbs", "adjectives"]]
        a.bar(x + (j - 1) * .25, values, width=.23, color=color, label=label)
    a.set_xticks(x, ["Nouns", "Verbs", "Adjectives\n(held out)"])
    a.set_ylabel("Mean raw DAT score")
    a.set_ylim(0, 105)
    a.set_title("Semantic distance increases", fontsize=15, color=INK, pad=20)
    a.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
    adjective = [counts_full["original"]["adjectives"], counts_lora["step1500"]["adjectives"], counts_full["step250"]["adjectives"]]
    per_word = [v["compatible_word_percent"] for v in adjective]
    per_list = [100 * v["joint_valid_lists"] / v["lists"] for v in adjective]
    for j, (word, lists, color) in enumerate(zip(per_word, per_list, colors)):
        b.bar(j - .17, word, width=.31, color=color)
        b.bar(j + .17, lists, width=.31, color=color, alpha=.4, hatch="//")
        b.text(j - .17, word + 2, f"{word:.1f}%", ha="center", fontsize=9, color=INK)
        b.text(j + .17, lists + 2, f"{lists:.1f}%", ha="center", fontsize=9, color=INK)
    b.set_xticks(x, labels)
    b.set_ylim(0, 117)
    b.set_ylabel("Adjective compliance (%)")
    b.set_title("Adjective following does not transfer", fontsize=15, color=INK, pad=20)
    b.text(.02, .95, "Solid: compatible words  ·  hatched: fully valid lists", transform=b.transAxes, fontsize=9, color=BASE)
    fig.suptitle("A high DAT score can hide a task failure", fontsize=22, weight="bold", x=.065, ha="left", y=.99, color=INK)
    fig.text(.065, .025, "32 responses per word class/model. WordNet any-sense audit; 10 requested words per list. Separate matched baseline runs.", fontsize=9, color=BASE)
    fig.subplots_adjust(top=.79, bottom=.18, left=.075, right=.98, wspace=.28)
    save(fig, "dat-and-validity")


def initial_writing_results():
    scores = load("writing-score-tables.json")
    with tarfile.open(ROOT / "results/raw/writing-transfer.tar.gz") as bundle:
        steer = json.load(bundle.extractfile("steer-metrics.json"))["models"]
    diagnostics = {}
    attributes = ["reading_difficulty", "textual_diversity", "text_length", "formality"]
    cutpoints = np.array([-1., -.5, -.2, -.1, .1, .2, .5, 1.])
    for name, values in steer.items():
        items = list(values["items"].values())
        source = np.array([[item[f"source_{a}"] for a in attributes] for item in items])
        output = np.array([[item[f"output_{a}"] for a in attributes] for item in items])
        # Same source-relative bins as the pinned upstream get_dist_from_source.
        dynamic = np.clip(source[..., None] + cutpoints, 0, 1)
        displacement = ((output[..., None] >= dynamic).sum(axis=2) - 4) / 7
        moving = np.linalg.norm(displacement, axis=1) > 0
        side_effects = np.array([item["side_effect_error"] for item in items])
        diagnostics[name] = {
            "n": len(items), "no_measured_movement": int((~moving).sum()),
            "no_measured_movement_percent": float(100 * (~moving).mean()),
            "moving_side_effect_error_mean": float(side_effects[moving].mean()),
        }
    (ROOT / "results/summary/steer-movement-diagnostics.json").write_text(
        json.dumps({"method": "Pinned SteerBench source-relative four-attribute bins; remaining in a bin does not imply identical text.",
                    "models": diagnostics}, indent=2) + "\n")
    return scores, diagnostics


def initial_writing_figure(scores, movement):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10), facecolor=BG)
    colors = [BASE, LORA, FULL]
    labels = ["Baseline", "LoRA1500", "Full250"]
    for ax in axes.flat:
        ax.set_facecolor(BG)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="y", color="#e8e8e2")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    a, b, c, d = axes.flat
    writer = scores["results/writingbench_critic"]["rows"]
    for i, (row, color) in enumerate(zip(writer, colors)):
        values = [row[3], row[5]]
        positions = np.arange(2) + (i - 1) * .24
        a.bar(positions, values, width=.22, color=color, label=labels[i])
        for x, y in zip(positions, values):
            a.text(x, y + .15, f"{y:.2f}", ha="center", fontsize=10, color=INK)
    a.set_xticks([0, 1], ["Overall", "Style"])
    a.set_ylim(0, 10.5)
    a.set_ylabel("Critic score (1–10, higher is better)")
    a.set_title("WritingBench critic scores", loc="left", fontsize=15, pad=15)

    steer = scores["results/steerbench"]["rows"]
    for i, (row, color) in enumerate(zip(steer, colors)):
        positions = np.arange(3) + (i - 1) * .24
        b.bar(positions, row[2:5], width=.22, color=color)
        for x, y in zip(positions, row[2:5]):
            b.text(x, y + .025, f"{y:.3f}", ha="center", fontsize=9, color=INK)
    b.set_xticks([0, 1, 2], ["Steering\nerror", "Miscalibration", "Side-effect\nerror"])
    b.set_ylim(0, 1.05)
    b.set_ylabel("Error (lower is better)")
    b.set_title("SteerBench aggregate errors", loc="left", fontsize=15, pad=15)

    for i, (name, color) in enumerate(zip(["original", "step1500", "step250"], colors)):
        item = movement[name]
        value = item["no_measured_movement_percent"]
        c.bar(i, value, width=.58, color=color)
        c.text(i, value + 1.5, f'{item["no_measured_movement"]}/{item["n"]}\n{value:.1f}%',
               ha="center", fontsize=11, color=INK)
    c.set_xticks([0, 1, 2], labels)
    c.set_ylim(0, 50)
    c.set_ylabel("Outputs in the source style bins (%)")
    c.set_title("SteerBench: no measured movement", loc="left", fontsize=15, pad=15)

    pairwise = scores["results/mazur_pairwise"]["rows"]
    d.grid(False)
    d.axvline(50, color=INK, linewidth=1, linestyle="--", alpha=.6)
    for i, (row, color) in enumerate(zip(pairwise, [LORA, FULL])):
        mean, low, high = row[3:6]
        d.errorbar(mean, i, xerr=[[mean-low], [high-mean]], fmt="o", color=color,
                   capsize=5, markersize=9, linewidth=2.4)
        d.text(72, i, f"{rounded(mean)}%", color=INK, va="center", fontsize=12, weight="bold")
    d.set_yticks([0, 1], ["LoRA1500", "Full250"])
    d.set_ylim(1.7, -.7)
    d.set_xlim(0, 100)
    d.set_xticks([0, 25, 50, 75, 100])
    d.set_xlabel("Win credit against baseline (%)")
    d.set_title("Phi4 story preference", loc="left", fontsize=15, pad=15)

    fig.suptitle("Initial writing and style evaluations", x=.07, ha="left", y=.985,
                 fontsize=23, weight="bold", color=INK)
    fig.legend(*a.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .94),
               ncol=3, frameon=False, fontsize=12)
    fig.text(.07, .052, "Phi4 chose the first-position story in 209/256 judgments (81.64%). Reversed-order agreement: LoRA 26.56%; Full 43.75%.",
             fontsize=10, color=INK)
    fig.text(.07, .026, "WritingBench: 64 requested cases/model. SteerBench: 128 edits/model. Phi4: 64 pairs/model; error bars are unadjusted 95% brief-cluster intervals.",
             fontsize=9.5, color=INK)
    fig.subplots_adjust(top=.84, bottom=.15, left=.075, right=.97, hspace=.6, wspace=.32)
    save(fig, "initial-writing-evaluations")


def initial_writing_tables(scores, movement):
    writer = scores["results/writingbench_critic"]["rows"]
    steer = scores["results/steerbench"]["rows"]
    ordered = [movement[name] for name in ["original", "step1500", "step250"]]
    lines = ["| Metric | Baseline | LoRA1500 | Full250 |", "| --- | ---: | ---: | ---: |"]
    for label, values in [
        ("WritingBench overall, higher better", [f"{row[3]:.2f}" for row in writer]),
        ("WritingBench style, higher better", [f"{row[5]:.2f}" for row in writer]),
        ("SteerBench steering error, lower better", [f"{row[2]:.4f}" for row in steer]),
        ("SteerBench miscalibration, lower better", [f"{row[3]:.4f}" for row in steer]),
        ("SteerBench side-effect error, lower better", [f"{row[4]:.4f}" for row in steer]),
        ("SteerBench outputs with no measured movement", [f'{row["no_measured_movement"]}/{row["n"]}' for row in ordered]),
        ("Side-effect error among moving outputs", [f'{row["moving_side_effect_error_mean"]:.4f}' for row in ordered]),
    ]:
        lines.append("| " + " | ".join([label, *values]) + " |")
    lines.extend(["", "WritingBench means use 62/62/63 fully scored cases; style uses all 64. SteerBench uses 128 edits/model. Moving-output subsets differ across models; that row is descriptive.", "",
                  "| Phi4 comparison | Win credit | 95% interval | Same winner after reversal | First-position preference |",
                  "| --- | ---: | --- | ---: | ---: |"])
    for row, diagnostic in zip(scores["results/mazur_pairwise"]["rows"], scores["results/mazur_judge_diagnostics"]["rows"]):
        lines.append(f'| {row[0].replace(" ", "")} vs baseline | {rounded(row[3])}% | [{rounded(row[4])}%, {rounded(row[5])}%] | {diagnostic[4]}/{diagnostic[5]} ({100*diagnostic[4]/diagnostic[5]:.2f}%) | {diagnostic[1]}/128 ({100*diagnostic[1]/128:.2f}%) |')
    lines.extend(["", "Both comparisons use 64 story pairs from 32 briefs, each judged in both orders. Intervals are unadjusted 95% brief-cluster bootstraps. Combined first-position preference is 209/256 (81.64%).", ""])
    (ROOT / "results/summary/initial-writing-table.md").write_text("\n".join(lines))


def tables():
    metrics = load("sol-results.json")["metrics"]
    lines = ["| Metric | Baseline | Full250 | Change | 95% interval | Full250 win credit |", "| --- | ---: | ---: | ---: | --- | ---: |"]
    for key, m in metrics.items():
        d = m["delta"]
        lines.append(f'| {key.replace("_", " ").capitalize()} | {m["baseline_mean"]:.2f} | {m["full250_mean"]:.2f} | {d["mean"]:+.2f} | [{d["ci95_low"]:+.2f}, {d["ci95_high"]:+.2f}] | {100*m["win_credit"]["mean"]:.2f}% |')
    (ROOT / "results/summary/sol-table.md").write_text("\n".join(lines) + "\n")
    rows = load("capabilities-aggregate.json")["rows"]
    lines = ["| Benchmark | Mode | N | Baseline | LoRA1500 | Full250 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for bench, mode, metric, base, lora, full, n in rows:
        lines.append(f"| {bench} | {mode.replace('_', ' ')} | {n} | {base:.2f}% | {lora:.2f}% | {full:.2f}% |")
    (ROOT / "results/summary/capabilities-table.md").write_text("\n".join(lines) + "\n")
    with (ROOT / "results/summary/capabilities-table.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["benchmark", "mode", "metric", "baseline", "lora1500", "full250", "n"])
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "svg.fonttype": "none"})
    poem_figure()
    judge_figure()
    dat_figure()
    initial, movement = initial_writing_results()
    initial_writing_figure(initial, movement)
    initial_writing_tables(initial, movement)
    tables()
    print("Created four figures in PNG/SVG and regenerated result tables.")


if __name__ == "__main__":
    main()
