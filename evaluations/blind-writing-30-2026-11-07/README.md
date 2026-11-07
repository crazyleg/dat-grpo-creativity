# Blind writing panel with 30 prompts

This package contains 15 short-poem prompts and 15 short-prose prompts, each
generated once by the baseline and Full250 models. The two outputs for each
prompt are assigned randomized A/B labels. The assignments vary across prompts.

## Files

- `prompts.jsonl` contains the supplied prompt IDs, topics and exact generation requests.
- `responses/poems/P01_A.txt` and `P01_B.txt`, and corresponding files through P15, contain the poems.
- `responses/prose/S01_A.txt` and `S01_B.txt`, and corresponding files through S15, contain the prose.
- `pairs.jsonl` contains all 30 pairs for automated evaluation.
- `output-stats.csv` records word counts, nonempty-line counts, completion tokens, stopping reasons and length compliance using the blinded labels.
- `protocol.json` records the common generation settings.
- `ground-truth.zip` contains the true model labels and paired seeds, password-protected using AES-256 ZIP encryption.
- `SHA256SUMS` records the released file checksums.

Both models use temperature 0.7, top-p 0.8, top-k 20 and a 2,048-token cap with
thinking disabled. Each supplied topic is preceded by its genre and length
instruction. Model output text is retained verbatim. Length and formatting
failures remain in the panel and are recorded in the statistics.

For judging, evaluate each pair in both presentation orders and lock the
ratings before requesting the password. The encrypted archive can then be
opened with 7-Zip or another application supporting AES-encrypted ZIP files.
Each prompt has its own randomized A/B model assignment.

## Poems with up to 16 lines

| ID | Prompt | Output A | Output B |
| --- | --- | --- | --- |
| P01 | A lighthouse keeper's last night before the light is automated. | [A](responses/poems/P01_A.txt) | [B](responses/poems/P01_B.txt) |
| P02 | A poem addressed to a word that has fallen out of use. | [A](responses/poems/P02_A.txt) | [B](responses/poems/P02_B.txt) |
| P03 | A city describes the people who never leave their street. | [A](responses/poems/P03_A.txt) | [B](responses/poems/P03_B.txt) |
| P04 | A love poem that never names the beloved or the feeling. | [A](responses/poems/P04_A.txt) | [B](responses/poems/P04_B.txt) |
| P05 | The sound a house makes when everyone is asleep. | [A](responses/poems/P05_A.txt) | [B](responses/poems/P05_B.txt) |
| P06 | A key that no longer fits any lock, in its own voice. | [A](responses/poems/P06_A.txt) | [B](responses/poems/P06_B.txt) |
| P07 | Learning to swim as an adult. | [A](responses/poems/P07_A.txt) | [B](responses/poems/P07_B.txt) |
| P08 | A map of a place that no longer exists. | [A](responses/poems/P08_A.txt) | [B](responses/poems/P08_B.txt) |
| P09 | An elegy for a phone number you still know by heart. | [A](responses/poems/P09_A.txt) | [B](responses/poems/P09_B.txt) |
| P10 | The first snow, without the words snow, white or cold. | [A](responses/poems/P10_A.txt) | [B](responses/poems/P10_B.txt) |
| P11 | A translator who is not sure the meaning survived. | [A](responses/poems/P11_A.txt) | [B](responses/poems/P11_B.txt) |
| P12 | The hour after a concert, when the hall is empty. | [A](responses/poems/P12_A.txt) | [B](responses/poems/P12_B.txt) |
| P13 | A river tries to explain time to a stone. | [A](responses/poems/P13_A.txt) | [B](responses/poems/P13_B.txt) |
| P14 | A grandmother's handwriting on a recipe card. | [A](responses/poems/P14_A.txt) | [B](responses/poems/P14_B.txt) |
| P15 | Waiting for test results, without naming the illness. | [A](responses/poems/P15_A.txt) | [B](responses/poems/P15_B.txt) |

## Short prose with 300 to 500 words

| ID | Prompt | Output A | Output B |
| --- | --- | --- | --- |
| S01 | A night-shift museum guard notices one painting has changed slightly every morning. | [A](responses/prose/S01_A.txt) | [B](responses/prose/S01_B.txt) |
| S02 | Two strangers share an umbrella and agree not to tell each other their names. | [A](responses/prose/S02_A.txt) | [B](responses/prose/S02_B.txt) |
| S03 | A man returns a library book forty years late and has to explain why. | [A](responses/prose/S03_A.txt) | [B](responses/prose/S03_B.txt) |
| S04 | A girl teaches her deaf grandfather a song using only her hands on his shoulders. | [A](responses/prose/S04_A.txt) | [B](responses/prose/S04_B.txt) |
| S05 | The last speaker of a language records a voicemail for no one. | [A](responses/prose/S05_A.txt) | [B](responses/prose/S05_B.txt) |
| S06 | A chess player realises mid-game that her opponent is letting her win. | [A](responses/prose/S06_A.txt) | [B](responses/prose/S06_B.txt) |
| S07 | A beekeeper has to tell the bees about a death in the family. | [A](responses/prose/S07_A.txt) | [B](responses/prose/S07_B.txt) |
| S08 | A ferry crossing is delayed and the passengers start telling each other lies. | [A](responses/prose/S08_A.txt) | [B](responses/prose/S08_B.txt) |
| S09 | A clockmaker is asked to repair a clock that runs backwards on purpose. | [A](responses/prose/S09_A.txt) | [B](responses/prose/S09_B.txt) |
| S10 | A woman finds a letter she wrote to herself at twelve and does not remember writing it. | [A](responses/prose/S10_A.txt) | [B](responses/prose/S10_B.txt) |
| S11 | A radio host realises the same caller has phoned in every night for ten years. | [A](responses/prose/S11_A.txt) | [B](responses/prose/S11_B.txt) |
| S12 | A family spends the last evening in their sold house without furniture. | [A](responses/prose/S12_A.txt) | [B](responses/prose/S12_B.txt) |
| S13 | An astronaut's child has to describe her mother's job at school. | [A](responses/prose/S13_A.txt) | [B](responses/prose/S13_B.txt) |
| S14 | A street musician plays a song a passer-by swears he wrote and never published. | [A](responses/prose/S14_A.txt) | [B](responses/prose/S14_B.txt) |
| S15 | A cartographer is hired to map a village that appears only in fog. | [A](responses/prose/S15_A.txt) | [B](responses/prose/S15_B.txt) |
