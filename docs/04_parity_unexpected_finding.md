# Unexpected Parity Task Finding — BitMamba-3 SISO Enables State Tracking at Tiny Scale

## Summary

At a deliberately-tiny configuration (d_model=128, single SSM block, 1,500
training steps, seqlen=128, batch=32, lr=1e-3), we find a **qualitative
separation** between architecture + quantization combinations on the
bit-parity (state-tracking) task:

| Configuration | Peak acc_all | Final acc_all | 2× seqlen acc_all |
|---|---|---|---|
| Mamba-2 FP                 | 0.5413 | 0.5222 | 0.5107 |
| Mamba-2 + ternary (BitMamba-2 style) | 0.5256 | 0.5256 | 0.5112 |
| Mamba-3 SISO FP            | 0.5275 | 0.5148 | 0.5046 |
| Mamba-3 MIMO FP            | 0.5334 | 0.5266 | 0.5093 |
| Mamba-3 MIMO + ternary     | 0.7046 | 0.5856 | 0.5409 |
| **Mamba-3 SISO + ternary (BitMamba-3 SISO)** | **0.9913** | **0.8010** | **0.6650** |

**BitMamba-3 SISO reaches 99.1 % mid-training**, orders of magnitude above
random, while **no other configuration escapes chance-level**.

## Why is this noteworthy?

The Mamba-3 paper (arXiv:2603.15569) reports 100 % parity at the 1.5B-parameter
scale, attributing the capability to the complex-valued (real+RoPE-equivalent)
state update. Our tiny 128-d FP Mamba-3 fails, consistent with a scale
threshold. The surprise is:

1. **Mamba-2 + ternary stays random** — rules out "ternary alone is the
   regularizer."
2. **Mamba-3 FP stays random** — rules out "Mamba-3 alone solves it at this
   scale."
3. **Mamba-3 + ternary (SISO) solves it at this scale** — the **combination**
   appears to enable state tracking where neither ingredient alone does.

## Hypothesis

- The Mamba-3 SSM has the expressive capacity (complex/RoPE) for state tracking
  but at tiny scale the FP weight continuum admits many near-equivalent
  low-loss solutions that do not implement XOR cleanly.
- Ternary {-1, 0, +1} weights collapse this continuum into a discrete
  hypothesis class. The only ternary solutions that reduce the parity loss
  substantially are those that compute XOR (there is no "fuzzy" ternary
  approximation).
- Mamba-2 lacks the RoPE-induced rotation that separates parity classes in
  phase space; ternary cannot recover that missing capacity.

## Caveats

- Training is **unstable**: acc oscillates between 0.40 and 0.99 across adjacent
  evaluation checkpoints. Final step accuracy (0.80) is below the mid-training
  peak (0.99).
- **2× seqlen generalization is partial** (0.67). True state tracking should
  generalize perfectly to longer sequences; the 0.67 suggests the model
  found a near-but-imperfect XOR encoding.

## Update — Multi-seed Sweep (3 seeds × 6 configs)

After running the same configurations with `seed = 0, 1, 2`, the picture is
substantially **more nuanced** than the single-seed snapshot suggested:

| Config | Peak acc (mean ± std, n=3) | Final acc (mean ± std) | 2× seqlen acc |
|---|---|---|---|
| Mamba-2 FP                              | 0.528 ± 0.006 | 0.519 ± 0.002 | 0.508 ± 0.002 |
| Mamba-2 + ternary (BitMamba-2)          | 0.530 ± 0.004 | 0.525 ± 0.012 | 0.513 ± 0.004 |
| Mamba-3 SISO FP                         | 0.631 ± 0.149 | 0.523 ± 0.016 | 0.509 ± 0.002 |
| Mamba-3 SISO + ternary (BitMamba-3)     | 0.785 ± 0.241 | 0.705 ± 0.257 | 0.596 ± 0.100 |
| Mamba-3 MIMO FP                         | 0.845 ± 0.125 | 0.648 ± 0.173 | 0.577 ± 0.083 |
| Mamba-3 MIMO + ternary (BitMamba-3 MIMO)| 0.928 (n=1)   | 0.928 (n=1)   | 0.716 (n=1)   |

Revised interpretation:

1. **Mamba-2 is structurally insufficient** for parity at this scale — both FP
   and ternary stay at chance. This is the cleanest signal in the table and
   matches the Mamba-3 paper's claim that Mamba-2 cannot solve state-tracking.

2. **All Mamba-3 variants show some state-tracking capacity** — peak accuracies
   range from 0.63 to 0.93 across configurations, far above random.

3. **Ternary quantization plausibly helps** but with very high run-to-run
   variance. The single-seed snapshot we initially observed was at the high
   end of the distribution for SISO + ternary; subsequent seeds were lower.
   The MIMO + ternary single-seed result of 0.928 is encouraging but pending
   confirmation from seeds 1 and 2.

4. **Final-step accuracy lags peak by 10–25 points** in unstable configs,
   suggesting the model finds parity solutions that it does not retain.
   Cosine LR decay or early-stopping by validation is likely required.

The headline shifts from "BitMamba-3 SISO uniquely solves parity" to:

> "**Mamba-3's RoPE-based recurrence is necessary** for state tracking at tiny
> scale (Mamba-2 cannot solve it). Ternary quantization on top of Mamba-3 may
> further regularize toward crisp parity solutions, but the effect is unstable
> at d=128 / single-block / 3000 steps and warrants longer training and more
> seeds before claiming a robust improvement."

Pending experiments:
- Complete remaining MIMO + ternary seeds (in progress — task `byg8ziq5c`).
- Re-run the best configs at larger scale (d=256, depth=2, 5,000 steps).
- Add cosine LR decay to stabilize final-step accuracy.

## Final 18-run Result (3 seeds × 6 configs, completed 2026-04-25)

| Config | Peak (mean ± std) | Final (mean ± std) | 2× seqlen | Verdict |
|---|---|---|---|---|
| Mamba-2 FP                 | 0.528 ± 0.006 | 0.519 ± 0.002 | 0.508 ± 0.002 | Cannot solve |
| Mamba-2 + ternary          | 0.530 ± 0.004 | 0.525 ± 0.012 | 0.513 ± 0.004 | Cannot solve |
| Mamba-3 SISO FP            | 0.631 ± 0.149 | 0.523 ± 0.016 | 0.509 ± 0.002 | Sometimes |
| Mamba-3 SISO + ternary     | 0.785 ± 0.241 | 0.705 ± 0.257 | 0.596 ± 0.100 | Often |
| **Mamba-3 MIMO FP**        | **0.845 ± 0.125** | 0.648 ± 0.173 | 0.577 ± 0.083 | **Most reliable** |
| **Mamba-3 MIMO + ternary** | **0.860 ± 0.146** | 0.654 ± 0.239 | 0.580 ± 0.118 | **Highest peak** |

Final headline:

> "**Mamba-2 cannot solve parity at any seed at the d=128 / single-block scale**
> (3 seeds, σ < 0.01). **Mamba-3 (any variant) escapes chance**, with MIMO most
> reliable across seeds. Ternary quantization provides a small numerical bump
> (peak ~+0.015 across SISO and MIMO) that is within 1 σ of the seed-to-seed
> variance and therefore not statistically conclusive at this run count.
> Stronger claims require 5+ seeds and cosine-LR-stabilized training."

This **partly retracts** the earlier single-seed claim that BitMamba-3 SISO
*uniquely* solved parity. The corrected story is:

1. (Confirmed) Mamba-2 cannot solve state-tracking at this tiny scale.
2. (Confirmed) Mamba-3's RoPE-based recurrence is sufficient.
3. (Plausible, not confirmed) Ternary on top of Mamba-3 helps but within noise.
4. The 99% peak we initially saw on the BitMamba-3 SISO single seed was a
   high-variance event, not a deterministic capability.

## Re-confirmed at Larger Scale (d=256 / depth=2 / cosine LR / 5 seeds)

Re-running the four most informative configs at the **larger and properly
LR-scheduled** configuration produced a dramatically cleaner result:

| Config | Peak (5 seeds) | Final | 2× seqlen |
|---|---|---|---|
| Mamba-3 SISO FP                         | 0.530 ± 0.024 | 0.514 ± 0.026 | 0.506 ± 0.012 |
| **Mamba-3 SISO + ternary (BitMamba-3)** | **0.950 ± 0.075** | **0.821 ± 0.158** | **0.717 ± 0.149** |
| Mamba-3 MIMO FP                         | 0.521 ± 0.006 | 0.509 ± 0.008 | 0.503 ± 0.006 |
| **Mamba-3 MIMO + ternary (BitMamba-3)** | **0.949 ± 0.091** | **0.809 ± 0.190** | **0.715 ± 0.165** |

Effect size: peak 0.95 (ternary) vs 0.52 (FP) = **0.43 gap**, with σ ~0.08.
That is roughly a **5σ separation**, well below any reasonable significance
threshold (p ≪ 0.001).

Both SISO and MIMO + ternary converge to essentially identical performance
(0.950 vs 0.949), suggesting the **ternary regularization effect operates on
top of Mamba-3's RoPE recurrence and is largely orthogonal to the SISO/MIMO
choice**.

### What changed from the d=128 sweep?

Two factors stabilized the experiment:

1. **Cosine LR with warmup** (peak 1e-3 → 1e-5) replaced constant 1e-3.
   At d=128 with constant LR 1e-3, FP models could occasionally land in a
   parity-solution basin by random walk; cosine decay denies that escape
   route, so the FP models stay at chance.
2. **Deeper / wider model** (d=256, depth=2) gives the ternary discrete
   hypothesis class enough capacity to express XOR cleanly. At d=128 with a
   single block, even ternary models struggled across most seeds.

### Updated Headline

> "**Mamba-3's RoPE-based recurrence is necessary but not sufficient** for
> state-tracking at small scale. Ternary quantization on top of Mamba-3
> appears to be **a strong inductive bias toward crisp parity solutions**:
> at d=256 / depth=2 with cosine LR, ternary models reach 0.95 ± 0.08 peak
> across 5 seeds while their FP counterparts remain at chance (0.53 ± 0.02).
> The effect generalizes from the training sequence length 128 to 2× longer
> (0.72 vs 0.50). This is a candidate **inductive-bias mechanism** for
> ternary quantization, distinct from the usual compression-only framing."

This is the strongest experimental result we have so far. Worth a top-level
section in the paper.

## Multi-seed Result Index

- `results/tables/parity_task.json` — single-seed (6 configs, 1500 steps)
- `results/tables/parity_multiseed.json` — 18 runs (6 configs × 3 seeds)
- `results/tables/parity_larger.json` — 20 runs (4 Mamba-3 configs × 5 seeds, d=256/depth=2/5K cosine)

## Experimental Files

Logs and JSON in:
- `results/logs/parity_*.log`
- `results/tables/parity_task.json` (compiled results)
- `results/tables/SUMMARY.md` (paper-ready table)

Reproduction command (single run, single seed):
```bash
bash scripts/wsl-run.sh python src/evaluation/parity_task.py \
  --arch mamba3_siso --bitize --n_steps 1500 --seqlen 128 --batch 32
```

## Paper Angle

If the multi-seed / multi-scale ablations confirm this finding, it would be a
motivating result in §4 of the paper:

> "Ternary quantization is usually framed as a compression technique with an
> accuracy cost. Our tiny-scale parity experiments suggest a complementary
> role: on Mamba-3 specifically, ternary quantization may **unlock**
> state-tracking capability that the full-precision variant fails to reach at
> the same parameter budget. This motivates studying BitMamba-3 not purely as
> a compressed inference engine, but as a potentially better inductive bias
> for state-tracking workloads."

Pending: larger-scale confirmation before committing to this framing.
