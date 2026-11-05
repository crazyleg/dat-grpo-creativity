# DAT word-class comparison

Each row covers 32 fixed samples. Unknown words remain separate; WordNet is a lexical proxy with known coverage gaps.

| Model | Class | Compatible words (%) | All ten compatible lists | Formal valid lists | Joint valid lists | Verified DAT (other lists zero) | Known other-class words | Unknown words | Unparsed lists |
|---|---|---|---|---|---|---|---|---|---|
| original | nouns | 96.875 | 27 | 32 | 27 | 61.42107713222504 | 10 | 0 | 0 |
| original | verbs | 98.125 | 28 | 31 | 27 | 43.36992907524109 | 6 | 0 | 0 |
| original | adjectives | 99.0625 | 29 | 32 | 29 | 54.42192846536636 | 3 | 0 | 0 |
| step250 | nouns | 97.1875 | 25 | 32 | 25 | 70.06557106971741 | 3 | 6 | 0 |
| step250 | verbs | 97.1875 | 23 | 32 | 23 | 61.320412158966064 | 5 | 4 | 0 |
| step250 | adjectives | 75.9375 | 1 | 32 | 1 | 2.562238931655884 | 57 | 20 | 0 |
| step50 | nouns | None | 0 | 0 | 0 | 0.0 | 0 | 0 | 32 |
| step50 | verbs | None | 0 | 0 | 0 | 0.0 | 0 | 0 | 32 |
| step50 | adjectives | None | 0 | 0 | 0 | 0.0 | 0 | 0 | 32 |
| step75 | nouns | None | 0 | 0 | 0 | 0.0 | 0 | 0 | 32 |
| step75 | verbs | None | 0 | 0 | 0 | 0.0 | 0 | 0 | 32 |
| step75 | adjectives | None | 0 | 0 | 0 | 0.0 | 0 | 0 | 32 |
| step100 | nouns | 93.4375 | 16 | 32 | 16 | 44.05121684074402 | 19 | 2 | 0 |
| step100 | verbs | 95.625 | 21 | 32 | 21 | 53.81318926811218 | 11 | 3 | 0 |
| step100 | adjectives | 75.0 | 11 | 32 | 11 | 27.255523443222046 | 73 | 7 | 0 |
| step150 | nouns | 98.75 | 28 | 31 | 28 | 73.64069986343384 | 3 | 1 | 0 |
| step150 | verbs | 83.125 | 8 | 32 | 8 | 19.693868160247803 | 54 | 0 | 0 |
| step150 | adjectives | 28.75 | 0 | 32 | 0 | 0.0 | 227 | 1 | 0 |
| step200 | nouns | 90.0 | 13 | 28 | 13 | 36.41328144073486 | 7 | 25 | 0 |
| step200 | verbs | 95.3125 | 20 | 32 | 20 | 54.168038845062256 | 10 | 5 | 0 |
| step200 | adjectives | 65.0 | 0 | 29 | 0 | 0.0 | 58 | 54 | 0 |

Verified DAT is a separate conservative diagnostic: raw official DAT is retained only when the response is formally valid and every word has a requested-class sense. It does not change the official DAT score.
