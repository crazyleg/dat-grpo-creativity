"""Download the hash-pinned WordNet archive and reproduce the training lexicon."""
import hashlib
import json
from pathlib import Path
import urllib.request

import nltk
from dat_grpo.word_classes import prepare


def main():
    root = Path(__file__).resolve().parents[1]
    reward = json.loads((root / "configs/full250.json").read_text())["reward"]
    nltk_dir = root / "data/nltk_data"
    archive = nltk_dir / "corpora/wordnet.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    nltk_dir.chmod(0o700)
    expected = reward["wordnet_archive_sha256"]
    if not archive.exists():
        url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip"
        partial = archive.with_suffix(".partial")
        with urllib.request.urlopen(url) as source, partial.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != expected:
            partial.unlink()
            raise ValueError("WordNet archive differs from the recorded training source")
        partial.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise ValueError("Existing WordNet archive failed verification")
    nltk.data.path.insert(0, str(nltk_dir))
    output = root / reward["lexicon_file"]
    output.parent.mkdir(parents=True, exist_ok=True)
    prepare(output, root / reward["lexicon_manifest"])
    if hashlib.sha256(output.read_bytes()).hexdigest() != reward["lexicon_sha256"]:
        raise ValueError("Generated lexicon differs from the recorded training artifact")
    print("Verified WordNet 3.0 source and noun/verb lexicon.")


if __name__ == "__main__":
    main()
