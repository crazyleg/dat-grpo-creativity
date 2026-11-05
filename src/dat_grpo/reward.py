"""Frozen official DAT scoring with a strict, bounded response format."""

import json
import hashlib
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .common import sha256_file, utc_now
from .vendor.dat.dat import Model

WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
VENDOR = Path(__file__).parent / "vendor" / "dat"


class FrozenDAT(Model):
    """Reuse the original validation, distance, and scoring methods unchanged."""

    def __init__(self, cache_path, manifest_path):
        self.manifest = json.loads(Path(manifest_path).read_text())
        expected = {
            "cache_sha256": sha256_file(cache_path),
            "dictionary_sha256": sha256_file(VENDOR / "words.txt"),
            "dat_source_sha256": sha256_file(VENDOR / "dat.py"),
        }
        for key, actual in expected.items():
            if self.manifest.get(key) != actual:
                raise ValueError(f"DAT provenance mismatch: {key}")
        with np.load(cache_path, allow_pickle=False) as data:
            words = data["words"]
            vectors = data["vectors"].copy()
        if vectors.shape != (len(words), 300) or not np.isfinite(vectors).all():
            raise ValueError("DAT cache must contain finite 300-dimensional vectors")
        if len(set(words)) != len(words) or (np.linalg.norm(vectors, axis=1) == 0).any():
            raise ValueError("DAT cache has duplicate words or zero vectors")
        vectors.flags.writeable = False
        self.vectors = dict(zip(words.tolist(), vectors))


@dataclass(frozen=True)
class Outcome:
    score: float | None
    reward: float
    words: list[str]
    selected_words: list[str]
    excluded_words: list[str]
    reason: str | None


def score_completion(text: str, model: Model) -> Outcome:
    """Reward is official DAT / 100; invalid responses receive zero.

    The JSON and ten-single-word requirement is a formatting gate. It does not
    replace official DAT dictionary validation and does not enforce word class.
    """
    try:
        words = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return Outcome(None, 0.0, [], [], [], "invalid_json")
    if not isinstance(words, list) or len(words) != 10:
        return Outcome(None, 0.0, [], [], [], "expected_ten_words")
    if any(not isinstance(word, str) or not WORD.fullmatch(word.strip()) for word in words):
        return Outcome(None, 0.0, [], [], [], "expected_single_english_words")
    words = [word.strip() for word in words]
    selected, excluded = [], []
    for word in words:
        valid = model.validate(word)
        if valid and valid not in selected:
            selected.append(valid)
        else:
            excluded.append(word)
    score = model.dat(words, minimum=7)
    if score is None:
        return Outcome(None, 0.0, words, selected, excluded, "fewer_than_seven_valid_unique_words")
    score = float(score)
    if not math.isfinite(score):
        raise ValueError("Non-finite reward from the frozen DAT scorer")
    return Outcome(score, score / 100.0, words, selected[:7], excluded, None)


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for message in completion:
            content = message.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(p.get("text", "") for p in content if isinstance(p, dict))
        return "".join(parts)
    return ""


class DATReward:
    __name__ = "dat_reward"

    def __init__(self, model, log_path, group_size):
        self.model = model
        self.log_path = Path(log_path)
        self.group_size = group_size
        self.pending = []
        self.calls = 0

    def __call__(self, completions, trainer_state=None, **kwargs):
        records = []
        step = getattr(trainer_state, "global_step", 0)
        for index, completion in enumerate(completions):
            text = completion_text(completion)
            result = score_completion(text, self.model)
            records.append({
                "time": utc_now(), "optimizer_step_before_update": step,
                "reward_call": self.calls, "group": index // self.group_size,
                "completion": text, **asdict(result),
            })
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as log:
            for record in records:
                log.write(json.dumps(record, allow_nan=False) + "\n")
        self.pending.extend(records)
        self.calls += 1
        return [record["reward"] for record in records]

    def drain(self):
        records, self.pending = self.pending, []
        return records


@dataclass(frozen=True)
class ClassOutcome(Outcome):
    word_class: str
    base_reward: float
    class_penalty: float
    invalid_penalty: float
    class_correct_words: int
    class_wrong_words: int
    unverified_nv_words: int
    ambiguous_nv_words: int
    word_checks: list[dict]


def score_class_completion(text, model, verifier, requested, penalty_per_word=.2, invalid_reward=-2.):
    if requested not in ('nouns', 'verbs'):
        raise ValueError('Adjective DAT is held out; only noun/verb rewards are supported')
    if not math.isfinite(penalty_per_word) or penalty_per_word <= 0 or not math.isfinite(invalid_reward) or invalid_reward > -10*penalty_per_word:
        raise ValueError('Invalid output must not evade the word-class penalty')
    result = score_completion(text, model)
    checks = [{'word': word, 'senses': list(verifier.senses(word)),
               'correct': verifier.accepts(word, requested)} for word in result.words]
    correct = sum(check['correct'] for check in checks)
    wrong = 10-correct
    penalty = penalty_per_word*wrong
    reward = result.reward-penalty if result.score is not None else invalid_reward
    return ClassOutcome(score=result.score, reward=reward, words=result.words,
        selected_words=result.selected_words, excluded_words=result.excluded_words, reason=result.reason,
        word_class=requested, base_reward=result.reward, class_penalty=penalty,
        invalid_penalty=result.reward-penalty-reward, class_correct_words=correct, class_wrong_words=wrong,
        unverified_nv_words=sum(not check['senses'] for check in checks)+(10-len(checks)),
        ambiguous_nv_words=sum(len(check['senses'])==2 for check in checks), word_checks=checks)


class ConditionalDATReward(DATReward):
    __name__ = 'dat_class_reward'

    def __init__(self, model, log_path, group_size, verifier, penalty_per_word, invalid_reward):
        super().__init__(model, log_path, group_size)
        self.verifier = verifier
        self.penalty_per_word = penalty_per_word
        self.invalid_reward = invalid_reward

    def __call__(self, completions, word_class=None, prompt_id=None, strict_class=None,
                 trainer_state=None, prompts=None, **kwargs):
        count = len(completions)
        if not count or count % self.group_size:
            raise ValueError('Conditional rewards require complete generation groups')
        columns = {'word_class':word_class, 'prompt_id':prompt_id, 'strict_class':strict_class}
        for name, values in columns.items():
            if not isinstance(values, (list, tuple)) or len(values) != count:
                raise ValueError(f'Missing or misaligned per-completion {name} metadata')
        if prompts is not None and len(prompts) != count:
            raise ValueError('Prompt/completion metadata alignment differs')
        for start in range(0, count, self.group_size):
            if any(len(set(values[start:start+self.group_size])) != 1 for values in columns.values()):
                raise ValueError('Each GRPO group must share its class, wording, and reminder')
        records = []
        for index, completion in enumerate(completions):
            requested = word_class[index]
            text = completion_text(completion)
            outcome = score_class_completion(text, self.model, self.verifier, requested,
                self.penalty_per_word, self.invalid_reward)
            row = {'time':utc_now(), 'optimizer_step_before_update':getattr(trainer_state,'global_step',0),
                'reward_call':self.calls, 'group':index//self.group_size, 'completion':text,
                'prompt_id':prompt_id[index], 'strict_class':strict_class[index], **asdict(outcome)}
            if prompts is not None:
                row['prompt_sha256'] = hashlib.sha256(str(prompts[index]).encode()).hexdigest()
            records.append(row)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open('a') as stream:
            for row in records:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
        self.pending.extend(records)
        self.calls += 1
        return [row['reward'] for row in records]
