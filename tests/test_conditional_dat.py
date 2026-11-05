import collections
import copy
import json
import tempfile
import unittest
from pathlib import Path

from dat_grpo.common import sha256_file, write_json
from dat_grpo.conditional_task import prompt_rows, restore_class_history
from dat_grpo.reward import ConditionalDATReward, score_class_completion
from dat_grpo.word_classes import WordClasses
from test_reward import WORDS, model_with_vectors

ROOT = Path(__file__).resolve().parents[1]


class Verifier:
    def senses(self, word):
        return ('nouns',) if word in WORDS[:7] else ('verbs',) if word in WORDS[7:] else ()

    def accepts(self, word, requested):
        return requested in self.senses(word)


class ConditionalRewardTests(unittest.TestCase):
    def test_six_generation_groups_keep_class_metadata_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            reward=ConditionalDATReward(model_with_vectors(),Path(folder)/'rollouts.jsonl',6,Verifier(),.2,-2)
            text=json.dumps(WORDS)
            values=reward([text]*12,word_class=['nouns']*6+['verbs']*6,
                prompt_id=['a']*12,strict_class=[False]*12,prompts=['noun task']*6+['verb task']*6)
            records=reward.drain()
            self.assertEqual(len(values),12)
            self.assertEqual([r['group'] for r in records],[0]*6+[1]*6)
            self.assertEqual([r['word_class'] for r in records],['nouns']*6+['verbs']*6)
            self.assertTrue(all(r['class_wrong_words']==3 for r in records[:6]))
            self.assertTrue(all(r['class_wrong_words']==7 for r in records[6:]))

    def test_class_penalty_checks_last_three_words_even_when_dat_is_unchanged(self):
        result = score_class_completion(json.dumps(WORDS), model_with_vectors(), Verifier(), 'nouns')
        self.assertEqual(result.score, 100)
        self.assertEqual(result.class_correct_words, 7)
        self.assertAlmostEqual(result.class_penalty, .6)
        self.assertAlmostEqual(result.reward, .4)
        self.assertEqual(result.invalid_penalty, 0)

    def test_all_wrong_can_be_negative_and_malformed_cannot_evade_penalties(self):
        class Wrong(Verifier):
            def senses(self, word):return ('verbs',)
        wrong=score_class_completion(json.dumps(WORDS),model_with_vectors(),Wrong(),'nouns')
        invalid=score_class_completion('not json',model_with_vectors(),Wrong(),'nouns')
        self.assertEqual(wrong.reward,-1)
        self.assertEqual(invalid.reward,-2)
        self.assertEqual(invalid.class_wrong_words,10)
        self.assertLess(invalid.reward,wrong.reward)

    def test_unverified_words_are_penalized_and_holdout_class_is_rejected(self):
        class Unverified(Verifier):
            def senses(self, word):return ()
        value=score_class_completion(json.dumps(WORDS),model_with_vectors(),Unverified(),'nouns')
        self.assertEqual(value.unverified_nv_words,10)
        self.assertEqual(value.reward,-1)
        with self.assertRaisesRegex(ValueError,'held out'):
            score_class_completion(json.dumps(WORDS),model_with_vectors(),Verifier(),'adjectives')

    def test_callback_requires_aligned_homogeneous_group_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            reward=ConditionalDATReward(model_with_vectors(),Path(folder)/'rollouts.jsonl',4,Verifier(),.2,-2)
            text=json.dumps(WORDS)
            kwargs={'word_class':['nouns']*4+['verbs']*4,'prompt_id':['a']*4+['b']*4,
                    'strict_class':[False]*4+[True]*4,'prompts':['noun prompt']*4+['verb prompt']*4}
            values=reward([text]*8,**kwargs)
            self.assertTrue(all(abs(v-.4)<1e-9 for v in values[:4]))
            self.assertTrue(all(abs(v+.4)<1e-9 for v in values[4:]))
            rows=reward.drain()
            self.assertEqual([r['group'] for r in rows],[0]*4+[1]*4)
            self.assertEqual(rows[4]['word_class'],'verbs')
            self.assertEqual(len(rows[0]['prompt_sha256']),64)
            bad=copy.deepcopy(kwargs);bad['word_class'][0]='verbs'
            with self.assertRaisesRegex(ValueError,'share'):
                reward([text]*8,**bad)
            with self.assertRaisesRegex(ValueError,'metadata'):
                reward([text]*8)


class ConditionalPromptTests(unittest.TestCase):
    def test_resume_diagnostics_exclude_future_and_incomplete_retry_groups(self):
        records = []
        for step in range(1, 61):
            records += [{'optimizer_step_before_update':step-1, 'word_class':'nouns', 'version':0}]*4
        # Replace the final completed update with a complete verb retry, then
        # leave an interrupted noun retry and a future group in the log.
        records += [{'optimizer_step_before_update':59, 'word_class':'verbs', 'version':1}]*4
        records += [{'optimizer_step_before_update':59, 'word_class':'nouns', 'version':2}]*2
        records += [{'optimizer_step_before_update':60, 'word_class':'verbs', 'version':3}]*4
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'rollouts.jsonl'
            path.write_text(''.join(json.dumps(record)+'\n' for record in records))
            windows, seen = restore_class_history(path, 60, 4)
            self.assertEqual(dict(seen), {'nouns':236, 'verbs':4})
            self.assertEqual(len(windows['nouns']), 100)
            self.assertEqual(len(windows['verbs']), 4)
            self.assertTrue(all(record['version']==1 for record in windows['verbs']))
            with self.assertRaisesRegex(ValueError, 'does not cover'):
                restore_class_history(path, 62, 4)

    def test_all_wordings_classes_and_optional_reminders_are_balanced(self):
        config=json.loads((ROOT/'configs/lora1500.json').read_text())
        rows=prompt_rows(config['task'])
        self.assertEqual(len(rows),1120)
        self.assertEqual(len({r['text'] for r in rows}),28)
        counts=collections.Counter((r['prompt_id'],r['word_class'],r['strict_class']) for r in rows)
        self.assertEqual(set(counts.values()),{40})
        self.assertEqual(set(r['word_class'] for r in rows),{'nouns','verbs'})
        self.assertTrue(all(r['word_class'] in r['text'] for r in rows))
        self.assertTrue(all('adjective' not in r['text'].lower() for r in rows))
        self.assertEqual(rows,prompt_rows(config['task']))
        config['task']['word_classes']=['nouns','adjectives']
        with self.assertRaisesRegex(ValueError,'hold out'):
            prompt_rows(config['task'])


class FrozenWordClassTests(unittest.TestCase):
    def test_inflections_ambiguity_unknowns_and_manifest_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'lexicon.json';manifest=Path(folder)/'manifest.json'
            write_json(path,{'format':'wordnet-nv-v1','classes':{
                'nouns':{'lemmas':['cat','mouse','run'],'exceptions':{'mice':['mouse']},'substitutions':[['s','']]},
                'verbs':{'lemmas':['run','eat'],'exceptions':{'ran':['run']},'substitutions':[['s','']]}}})
            write_json(manifest,{'lexicon_sha256':sha256_file(path)})
            verifier=WordClasses(path,manifest)
            self.assertTrue(verifier.accepts('Cats','nouns'))
            self.assertTrue(verifier.accepts('mice','nouns'))
            self.assertTrue(verifier.accepts('ran','verbs'))
            self.assertEqual(set(verifier.senses('run')),{'nouns','verbs'})
            self.assertEqual(verifier.senses('nonsenseword'),())
            path.write_text(path.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'hash'):
                WordClasses(path,manifest)
