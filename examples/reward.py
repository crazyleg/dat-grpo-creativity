"""Score an actual ten-word response with the training reward; no GPU needed."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from dat_grpo.reward import FrozenDAT, score_class_completion
from dat_grpo.word_classes import WordClasses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="word_class", choices=["nouns", "verbs"], required=True)
    parser.add_argument("--words", required=True, help="JSON array of ten words")
    parser.add_argument("--data-dir", type=Path, default=Path("data/dat"))
    args = parser.parse_args()
    scorer = FrozenDAT(args.data_dir / "glove840b-dat.npz", args.data_dir / "manifest.json")
    verifier = WordClasses(args.data_dir / "wordnet-nv.json", args.data_dir / "wordnet-nv-manifest.json")
    # DAT uses the first seven valid unique words. Class checks cover all ten.
    outcome = score_class_completion(
        args.words, scorer, verifier, args.word_class,
        penalty_per_word=0.2, invalid_reward=-2.0,
    )
    print(json.dumps(asdict(outcome), indent=2))


if __name__ == "__main__":
    main()
