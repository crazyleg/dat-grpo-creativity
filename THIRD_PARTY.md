# Third party sources

The original Divergent Association Task scorer and dictionary are vendored in
`src/dat_grpo/vendor/dat/`. Their MIT license, authorship, pinned commit and
file hashes are retained there. Source:
https://github.com/jayolson/divergent-association-task.

GloVe and WordNet data are downloaded separately; this repository does not
redistribute their embedding or lexical archives. Their upstream terms apply:
https://nlp.stanford.edu/projects/glove/ and
https://wordnet.princeton.edu/license-and-commercial-use.

Benchmark source material retains its upstream terms. Result bundles contain
generated responses, judgments, scores and selection metadata. Full upstream
question banks and prompt collections are not redistributed. Source revisions
and case identifiers are recorded so readers can retrieve the corresponding
inputs from their original projects.

The MIT license covers the original code in this repository. Model weights
have their own license and model card on Hugging Face.
