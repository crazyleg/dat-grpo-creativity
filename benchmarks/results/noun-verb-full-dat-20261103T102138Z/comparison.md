# DAT nouns, verbs and adjective holdout: full finetune snapshots snapshot comparison

| Model | Checkpoint step | no_think dat_nouns/reward_score (score) | no_think dat_nouns/reward_score change | no_think dat_verbs/reward_score (score) | no_think dat_verbs/reward_score change | no_think dat_adjectives/reward_score (score) | no_think dat_adjectives/reward_score change |
|---|---|---|---|---|---|---|---|
| original | 0 | 71.81 | 0.0 | 51.11 | 0.0 | 60.72 | 0.0 |
| step250 | 250 | 89.59 | 17.78 | 85.21 | 34.11 | 86.26 | 25.55 |
| step50 | 50 | 0.0 | -71.81 | 0.0 | -51.11 | 0.0 | -60.72 |
| step75 | 75 | 0.0 | -71.81 | 0.0 | -51.11 | 0.0 | -60.72 |
| step100 | 100 | 87.6 | 15.78 | 82.25 | 31.14 | 81.23 | 20.51 |
| step150 | 150 | 81.31 | 9.49 | 82.08 | 30.98 | 81.77 | 21.06 |
| step200 | 200 | 78.27 | 6.45 | 87.36 | 36.25 | 79.04 | 18.33 |

The same 96 fixed cases are used for every snapshot. Higher is better.
Changes are versus the original in the same mode, in percentage points for accuracy or DAT score points.
Empty cells are pending. Scores appear immediately after a model/mode completes.
Same32 fixed cases per class and sample seeds as the prior conditional LoRA panel. Fresh original baseline, then full250, then50/75/100/150/200. T=1, top_p=1, unrestricted top_k, no thinking,192 completion tokens,64 concurrent requests. Official DAT assigns0 to unscoreable outputs; lexical and joint validity reported separately.
Output budgets are local experiment limits. Truncation fractions and raw outputs are retained. Thinking is separated from the graded final answer.
This is a matched regression screen on the common BF16 unpacking of the NF4 training backbone, not a reproduction of leaderboard results.
The HumanEval canonical self-check has a known numerical reference failure on task 32; native test predicates are unchanged.
Paired bootstrap intervals are saved in results.json; small changes require follow-up.
