# Article and image assets

- [`post.md`](post.md) is the complete article with Markdown tables, links,
  captions and native LaTeX equations, suitable for GitHub rendering.
- [`substack.md`](substack.md) has the same text with equations rendered as
  images, for editors that do not render LaTeX. Upload the referenced PNGs
  when importing it into Substack. Review the editor's preview before posting.
- [`figures/`](figures/) contains PNG and editable SVG versions. The four main
  figures are the eternal-love sonnet pair, the DAT/validity comparison,
  the initial writing/style results and the paired creativity-score changes.
  Four additional images render the math.

`scripts/build_figures.py` and `scripts/build_substack.py` regenerate the assets.
The sonnet text comes from `results/samples/`; all score charts use the released
summary JSON. No new model generation or judge request is made by these scripts.

The source data and methods support increased judged creativity and imagery,
with lower coherence and instruction following. The article preserves those
tradeoffs and treats the author's informal human reading separately from the
paired judge analysis.
