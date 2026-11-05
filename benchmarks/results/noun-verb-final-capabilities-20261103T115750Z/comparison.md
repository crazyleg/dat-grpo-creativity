# Sequential snapshot benchmark campaign

| Benchmark | Mode | Metric | Baseline | LoRA 1500 | Full 250 | N |
|---|---|---|---|---|---|---|
| MMLU-Pro | no_think | accuracy_percent | 59.29 | 59.29 | 56.07 | 280 |
| MMLU-Pro | medium_think | accuracy_percent | 77.86 | 77.14 | 78.57 | 280 |
| HumanEval+ | no_think | humaneval/plus_pass_at_1 | 93.9 | 94.51 | 92.07 | 164 |
| HumanEval+ | medium_think | humaneval/plus_pass_at_1 | 93.29 | 93.9 | 93.9 | 164 |
| GSM8K | no_think | gsm8k/accuracy | 98.05 | 98.44 | 98.44 | 256 |
| GSM8K | medium_think | gsm8k/accuracy | 98.44 | 98.44 | 98.05 | 256 |
| IFEval | no_think | ifeval/strict_prompt_accuracy | 83.18 | 80.78 | 79.48 | 541 |

Accuracy/pass rates are percentages; DAT scores use the official 0-100 scale with invalid outputs zero.
Each benchmark finishes across all configured models before the next begins. Capability benchmarks run no-think first, then medium thinking. DAT uses the original training sampling without thinking.
Medium thinking uses an 8192-token total output limit. Truncation and formatting failures remain visible in each benchmark report. Small sampled differences are not proof of broad capability preservation.
All comparisons originate from the same frozen NF4 backbone; full snapshots contain the trained BF16 weights. These are matched local regression screens, not official leaderboard reproductions.
HumanEval uses unchanged native EvalPlus tests; its reference self-check has the documented numerical failure on task 32.

Individual benchmark reports:
- MMLU-Pro: [W&B](https://forge.coreweave.com/wandb/savsunenko-sasha/llm-creativity-grpo/runs/iel8xk02); local: ./benchmarks/results/noun-verb-final-capabilities-20261103T115750Z/00-mmlu_pro/comparison.md
- HumanEval+: [W&B](https://forge.coreweave.com/wandb/savsunenko-sasha/llm-creativity-grpo/runs/pwkbc3j0); local: ./benchmarks/results/noun-verb-final-capabilities-20261103T115750Z/01-humaneval/comparison.md
- GSM8K: [W&B](https://forge.coreweave.com/wandb/savsunenko-sasha/llm-creativity-grpo/runs/n5frm362); local: ./benchmarks/results/noun-verb-final-capabilities-20261103T115750Z/02-gsm8k/comparison.md
- IFEval: [W&B](https://forge.coreweave.com/wandb/savsunenko-sasha/llm-creativity-grpo/runs/7hmhmea6); local: ./benchmarks/results/noun-verb-final-capabilities-20261103T115750Z/03-ifeval/comparison.md
