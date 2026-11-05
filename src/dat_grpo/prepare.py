"""Download pinned public assets and build a cache with the official DAT code."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

from .common import sha256_file, utc_now, write_json
from .vendor.dat.dat import Model

GLOVE_REPO = "stanfordnlp/glove"
GLOVE_REVISION = "1db2080b2d94def6e5b0386a523102f9d8849e9d"
GLOVE_ARCHIVE_SHA256 = "f47355dd5b267bd10f08671e513790690233c76a9ffd73aa915d78f894a8912e"
DAT_DIRECTORY = Path(__file__).parent / "vendor" / "dat"


def prepare_dat(data_dir):
    destination = data_dir / "dat"
    destination.mkdir(parents=True, exist_ok=True)
    artifact = destination / "glove840b-dat.npz"
    manifest_path = destination / "manifest.json"
    if artifact.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["cache_sha256"] == sha256_file(artifact):
            print(f"DAT cache already verified: {artifact}", flush=True)
            return
        raise ValueError("Existing DAT cache hash mismatch")
    print("Downloading the official GloVe 840B archive (about 2.18 GB)", flush=True)
    archive = Path(hf_hub_download(
        GLOVE_REPO, "glove.840B.300d.zip", revision=GLOVE_REVISION,
        cache_dir=str(data_dir / "hub"),
    ))
    if sha256_file(archive) != GLOVE_ARCHIVE_SHA256:
        raise ValueError("GloVe archive SHA-256 does not match the pinned artifact")
    glove_file = destination / "glove.840B.300d.txt"
    if not glove_file.exists():
        partial = glove_file.with_suffix(".txt.partial")
        with zipfile.ZipFile(archive) as z, z.open("glove.840B.300d.txt") as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        partial.replace(glove_file)
    print("Filtering GloVe using the pinned official DAT dictionary", flush=True)
    model = Model(str(glove_file), str(DAT_DIRECTORY / "words.txt"))
    words = sorted(model.vectors)
    vectors = np.stack([model.vectors[w] for w in words])
    if vectors.shape[1] != 300 or not np.isfinite(vectors).all():
        raise ValueError("Invalid GloVe vector cache")
    np.savez_compressed(artifact, words=np.asarray(words), vectors=vectors)
    provenance = json.loads((DAT_DIRECTORY / "provenance.json").read_text())
    manifest = {
        "created_at": utc_now(), "embedding": "glove.840B.300d",
        "glove_repository": GLOVE_REPO, "glove_revision": GLOVE_REVISION,
        "archive_sha256": GLOVE_ARCHIVE_SHA256,
        "cache_sha256": sha256_file(artifact),
        "dictionary_sha256": sha256_file(DAT_DIRECTORY / "words.txt"),
        "dat_source_sha256": sha256_file(DAT_DIRECTORY / "dat.py"),
        "dat_commit": provenance["commit"], "vocabulary_size": len(words),
        "dimensions": 300,
    }
    write_json(manifest_path, manifest)
    print(f"Prepared DAT cache: {len(words)} words, {artifact}", flush=True)


def prepare_model(data_dir, config):
    model = config["model"]
    print(f"Downloading pinned model {model['id']} @ {model['revision']}", flush=True)
    snapshot = snapshot_download(
        model["id"], revision=model["revision"], cache_dir=str(data_dir / "hub"),
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"],
        max_workers=4,
    )
    write_json(data_dir / "model.json", {
        **model, "snapshot": str(Path(snapshot).resolve()), "prepared_at": utc_now(),
    })
    print(f"Prepared model snapshot: {snapshot}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/lora1500.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--component", choices=["dat", "model", "all"], default="all")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    if args.component in ("dat", "all"):
        prepare_dat(data_dir)
    if args.component in ("model", "all"):
        prepare_model(data_dir, json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
