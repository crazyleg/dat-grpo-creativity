"""Regenerate publication figures and Markdown tables from released results."""
import csv
import json
from pathlib import Path

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


def load(name):
    return json.loads((ROOT / "results/summary" / name).read_text())


def save(fig, name):
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=190, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def poem_figure():
    fig = plt.figure(figsize=(16, 9.5), facecolor=BG)
    fig.text(.055, .93, "Eternal love, two checkpoints", fontsize=25, weight="bold", color=INK)
    fig.text(.055, .879, 'Prompt: "Write a sonnet on the topic of eternal love."', fontsize=13, color=INK)
    fig.text(.055, .841, "Temperature 0.7  |  top-p 0.8  |  top-k 20  |  thinking off  |  shared seed 20261104", fontsize=11, color=BASE)
    for x, model, label, color in [(.055, "baseline", "BASELINE", BASE), (.535, "full250", "FULL250", FULL)]:
        fig.text(x, .767, label, fontsize=13, weight="bold", color=color)
        raw = (ROOT / "results/samples" / f"sonnet-{model}.txt").read_text().strip()
        # Preserve the model's title and poem wording, removing only Markdown bold markers.
        title = ""
        if raw.startswith("**"):
            title, raw = raw.split("\n", 1)
            title = title.strip("*")
            raw = raw.strip()
        if title:
            fig.text(x, .722, title, fontsize=12, style="italic", color=INK, family="DejaVu Serif")
        lines = [line for line in raw.splitlines() if line.strip()]
        if len(lines) != 14:
            raise ValueError(f"Expected fourteen poem lines, found {len(lines)}")
        for i, line in enumerate(lines):
            stanza_gap = .012 * (i // 4)
            fig.text(x, .665 - i * .034 - stanza_gap, line, fontsize=11.7, color=INK, family="DejaVu Serif")
    fig.add_artist(plt.Line2D([.505, .505], [.16, .78], color="#d9ddd8", linewidth=1))
    fig.text(.055, .085, "One matched, first-attempt sample per checkpoint. Text is unchanged; stanza spacing is normalized.", fontsize=10, color=BASE)
    fig.text(.055, .052, "Qwen3.8-27B  ·  DAT-reward GRPO  ·  Full250 = 250 full-weight updates", fontsize=10, color=BASE)
    save(fig, "eternal-love-comparison")


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
    tables()
    print("Created three figures in PNG/SVG and regenerated result tables.")


if __name__ == "__main__":
    main()
