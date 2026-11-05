"""Create a copy of the article with equations as uploadable images."""
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    equations = [
        ("dat-equation", [r"$D(W)=\frac{100}{21}\sum_{1\leq i<j\leq7}\left(1-\frac{e_i^\top e_j}{\|e_i\|\|e_j\|}\right)$"]),
        ("advantage-equation", [r"$A_i=\frac{R_i-\overline{R}}{\mathrm{std}(R)+\epsilon_A}$"]),
        ("grpo-objective", [r"$J(\theta)=\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_t\left[\min\left(\rho_{i,t}A_i,\mathrm{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)A_i\right)-\beta k_{i,t}\right]\right]$", r"$\rho_{i,t}=\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}{\pi_{\mathrm{old}}(y_{i,t}\mid x,y_{i,<t})},\qquad L=-J$"]),
        ("reward-equation", [r"$R(y,c)=D(y)/100-0.2\,N_{\mathrm{wrong\ class}}(y,c)$"]),
    ]
    for name, lines in equations:
        fig = plt.figure(figsize=(12, 1.6 if len(lines) == 1 else 2.6), facecolor="#fbfaf7")
        for i, line in enumerate(lines):
            fig.text(.5, .5 if len(lines) == 1 else .74-i*.48, line, ha="center", va="center", fontsize=17 if name == "grpo-objective" else 23, color="#243240")
        for suffix in ["png", "svg"]:
            fig.savefig(ROOT / "article/figures" / f"{name}.{suffix}", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    source = (ROOT / "article/post.md").read_text()
    blocks = list(re.finditer(r"\$\$.*?\$\$", source, flags=re.S))
    if len(blocks) != len(equations):
        raise ValueError("Article equation count changed")
    for match, (name, _) in reversed(list(zip(blocks, equations))):
        source = source[:match.start()] + f"![{name.replace('-', ' ').capitalize()}](figures/{name}.png)" + source[match.end():]
    source = source.replace("$-J$", "−J").replace("$k$", "k").replace("$\\varepsilon=0.1$", "ε = 0.1").replace("$\\beta=0.04$", "β = 0.04")
    (ROOT / "article/substack.md").write_text(source)
    print("Created equation images and article/substack.md.")


if __name__ == "__main__":
    main()
