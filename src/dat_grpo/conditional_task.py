"""Balanced noun/verb DAT prompts and conditional reward diagnostics."""

import collections
import json
import random
import statistics
from pathlib import Path

from .common import sha256_file


def prompt_rows(task):
    definitions = json.loads(Path(task['prompt_templates_file']).read_text())
    templates = definitions['templates']
    if len(templates) != 7 or len({t['id'] for t in templates}) != 7:
        raise ValueError('Expected seven distinct DAT wording templates')
    if set(task['word_classes']) != {'nouns', 'verbs'} or task['holdout_word_classes'] != ['adjectives']:
        raise ValueError('Training must use nouns/verbs and hold out adjective DAT')
    combinations = []
    for template in templates:
        if '{word_class}' not in template['text']:
            raise ValueError('Every template must explicitly condition on word class')
        for requested in task['word_classes']:
            for strict in [False, True]:
                prompt = template['text'].format(word_class=requested)
                if strict:
                    prompt += '\n'+definitions['strict_reminder'].format(word_class=requested)
                combinations.append({'text': prompt, 'word_class': requested,
                    'prompt_id': template['id'], 'strict_class': strict})
    if task['dataset_rows'] % len(combinations):
        raise ValueError('Dataset size must balance all template/class/reminder combinations')
    rows = combinations * (task['dataset_rows']//len(combinations))
    random.Random(task['prompt_seed']).shuffle(rows)
    return rows


def task_manifest(task, rows):
    return {'templates_sha256': sha256_file(task['prompt_templates_file']),
        'rows': len(rows), 'unique_rendered_prompts': len({r['prompt'] for r in rows}),
        'classes': dict(collections.Counter(r['word_class'] for r in rows)),
        'reminder': dict(collections.Counter(str(r['strict_class']) for r in rows)),
        'combinations': dict(collections.Counter(f"{r['prompt_id']}/{r['word_class']}/{r['strict_class']}" for r in rows)),
        'holdout_word_classes': task['holdout_word_classes']}


def class_metrics(records):
    if not records:
        return {}
    return {'base_reward_mean': statistics.mean(r['base_reward'] for r in records),
        'reward_mean': statistics.mean(r['reward'] for r in records),
        'penalty_mean': statistics.mean(r['class_penalty'] for r in records),
        'correct_word_fraction': sum(r['class_correct_words'] for r in records)/(10*len(records)),
        'all_ten_correct_fraction': statistics.mean(r['class_correct_words']==10 for r in records),
        'joint_valid_fraction': statistics.mean(r['score'] is not None and r['class_correct_words']==10 for r in records),
        'ambiguous_word_fraction': sum(r['ambiguous_nv_words'] for r in records)/(10*len(records)),
        'formal_valid_fraction': statistics.mean(r['score'] is not None for r in records)}


def restore_class_history(path, resume_step, group_size):
    """Restore diagnostics through the checkpoint, excluding uncommitted updates.

    A retried update may have multiple groups in the append-only rollout log.
    Use its last complete group, without counting an interrupted retry twice.
    """
    complete, pending = {}, {}
    with Path(path).open() as stream:
        for line in stream:
            record = json.loads(line)
            step = record['optimizer_step_before_update'] + 1
            if step > resume_step:
                continue
            group = pending.setdefault(step, [])
            group.append(record)
            if len(group) == group_size:
                complete[step] = group
                pending[step] = []
    if set(complete) != set(range(1, resume_step + 1)):
        raise ValueError('Completed rollout history does not cover the resumed checkpoint')
    windows = {name: collections.deque(maxlen=100) for name in ['nouns', 'verbs']}
    seen = collections.Counter()
    for step in sorted(complete):
        for record in complete[step]:
            requested = record['word_class']
            windows[requested].append(record)
            seen[requested] += 1
    return windows, seen
