"""Frozen WordNet noun/verb verification without a training-time NLTK dependency."""

import argparse
import functools
import json
import re
from pathlib import Path

from .common import sha256_file, write_json

CLASSES = {'nouns': 'n', 'verbs': 'v'}
SINGLE_WORD = re.compile(r'[a-z]+(?:-[a-z]+)*')


class WordClasses:
    def __init__(self, path, manifest_path):
        self.manifest = json.loads(Path(manifest_path).read_text())
        if sha256_file(path) != self.manifest['lexicon_sha256']:
            raise ValueError('Word-class lexicon hash differs from its manifest')
        self.data = json.loads(Path(path).read_text())
        if self.data['format'] != 'wordnet-nv-v1' or set(self.data['classes']) != set(CLASSES):
            raise ValueError('Expected the frozen noun/verb lexicon')
        self.lemmas = {name: set(value['lemmas']) for name, value in self.data['classes'].items()}

    @functools.lru_cache(maxsize=32768)
    def senses(self, word):
        word = word.strip().lower()
        found = []
        for name, value in self.data['classes'].items():
            # Mirrors WordNet _morphy: exception forms, otherwise one suffix pass,
            # plus the original word. Accept any requested-class sense.
            forms = value['exceptions'].get(word)
            if forms is None:
                forms = [word[:-len(old)]+new for old, new in value['substitutions'] if word.endswith(old)]
            if any(form in self.lemmas[name] for form in [word, *forms]):
                found.append(name)
        return tuple(found)

    def accepts(self, word, requested):
        if requested not in CLASSES:
            raise ValueError('Only noun and verb training classes are allowed')
        return requested in self.senses(word)


def prepare(output, manifest):
    import nltk
    from nltk.corpus import wordnet
    wordnet.ensure_loaded()
    archive = Path(str(nltk.data.find('corpora/wordnet.zip')))
    data = {'format': 'wordnet-nv-v1', 'wordnet_version': wordnet.get_version(), 'classes': {}}
    for name, pos in CLASSES.items():
        data['classes'][name] = {
            'pos': pos,
            'lemmas': sorted(word for word in wordnet.all_lemma_names(pos=pos) if SINGLE_WORD.fullmatch(word)),
            'exceptions': {word: forms for word, forms in wordnet._exception_map[pos].items()
                           if SINGLE_WORD.fullmatch(word)},
            'substitutions': wordnet.MORPHOLOGICAL_SUBSTITUTIONS[pos],
        }
    write_json(output, data)
    write_json(manifest, {'format': data['format'], 'wordnet_version': data['wordnet_version'],
        'source_archive_sha256': sha256_file(archive), 'lexicon_sha256': sha256_file(output),
        'method': 'Accept any noun/verb sense, including WordNet inflection rules; ambiguous words may satisfy both classes. Unverified words fail the class check. Adjective DAT is not a training task.',
        'lemma_counts': {name: len(value['lemmas']) for name, value in data['classes'].items()}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output, args.manifest)
