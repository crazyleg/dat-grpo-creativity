import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dat_grpo.common import sha256_file, write_json
from dat_grpo.reward import DATReward, FrozenDAT, VENDOR, score_completion
from dat_grpo.vendor.dat.dat import Model

WORDS = ["alfa", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]


def model_with_vectors(vectors=None):
    model = Model.__new__(Model)
    model.vectors = dict(zip(WORDS, np.eye(10, dtype=np.float32))) if vectors is None else vectors
    return model


class RewardTests(unittest.TestCase):
    def test_reward_uses_official_scale(self):
        result = score_completion(json.dumps(WORDS), model_with_vectors())
        self.assertAlmostEqual(result.score, 100)
        self.assertAlmostEqual(result.reward, 1)
        self.assertEqual(result.selected_words, WORDS[:7])

    def test_first_seven_are_used_even_if_later_words_increase_distance(self):
        vectors = {w: np.array([1, 0], dtype=np.float32) for w in WORDS[:7]}
        vectors.update({w: np.array([0, 1], dtype=np.float32) for w in WORDS[7:]})
        result = score_completion(json.dumps(WORDS), model_with_vectors(vectors))
        self.assertEqual(result.score, 0)

    def test_unknown_and_duplicate_words_use_the_official_redundancy(self):
        words = ["unknown", "ALFA", "alfa"] + WORDS[1:8]
        result = score_completion(json.dumps(words), model_with_vectors())
        self.assertEqual(result.selected_words, WORDS[:7])
        self.assertEqual(result.excluded_words, ["unknown", "alfa"])
        self.assertAlmostEqual(result.reward, 1)

    def test_insufficient_valid_words_get_zero_without_changing_denominator(self):
        result = score_completion(json.dumps(["alfa"] * 10), model_with_vectors())
        self.assertEqual(result.reward, 0)
        self.assertIsNone(result.score)
        self.assertEqual(result.reason, "fewer_than_seven_valid_unique_words")

    def test_prose_and_long_lists_cannot_game_the_reward(self):
        invalid = ["Here are ten words: " + json.dumps(WORDS), json.dumps(WORDS + ["extra"]),
                   json.dumps({"words": WORDS}), json.dumps(WORDS[:9]),
                   json.dumps(WORDS[:9] + [3]), json.dumps(WORDS[:9] + ["two words"])]
        for text in invalid:
            with self.subTest(text=text):
                self.assertEqual(score_completion(text, model_with_vectors()).reward, 0)

    def test_reward_callback_handles_both_trl_formats_and_writes_raw_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rollouts.jsonl"
            reward = DATReward(model_with_vectors(), path, 2)
            text = json.dumps(WORDS)
            values = reward([text, [{"role": "assistant", "content": text}]])
            self.assertEqual(values, [1, 1])
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["completion"], text)
            self.assertEqual(len(reward.drain()), 2)
            self.assertEqual(reward.drain(), [])

    def test_cache_matches_original_model_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "vectors.npz"
            vectors = np.eye(10, 300, dtype=np.float32)
            np.savez_compressed(cache, words=np.asarray(WORDS), vectors=vectors)
            manifest = root / "manifest.json"
            write_json(manifest, {
                "cache_sha256": sha256_file(cache),
                "dictionary_sha256": sha256_file(VENDOR / "words.txt"),
                "dat_source_sha256": sha256_file(VENDOR / "dat.py"),
            })
            frozen = FrozenDAT(cache, manifest)
            original = model_with_vectors(dict(zip(WORDS, vectors)))
            self.assertEqual(frozen.dat(WORDS), original.dat(WORDS))
            with self.assertRaises(ValueError):
                frozen.vectors[WORDS[0]][0] = 2
            with cache.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "cache_sha256"):
                FrozenDAT(cache, manifest)


if __name__ == "__main__":
    unittest.main()
