# Run 1: does training un-collapse the latent loop? (2026-09-18)

Qwen2.5-0.5B + LoRA r=16 (8.8M trainable), trained at k=4 for 300 steps on the
synthetic entailment-closure task. 321s on one T4, fp16. Final loss 0.0002.

## Result 1 — training did NOT un-collapse the trajectory

```
untrained cos(z_k, z_k+1): +0.963 +0.966 +0.979 +0.985 +0.987 +0.988 +0.987
trained   cos(z_k, z_k+1): +0.929 +0.972 +0.984 +0.990 +0.992 +0.993 +0.993
```

Training made the fixed point *tighter*, not looser. The first step moves more
(0.929 vs 0.963) but convergence is faster and lands higher (0.993 vs 0.987).
The latent loop still stops doing new work after ~4 steps.

## Result 2 — the k-curve saturates exactly where the trajectory collapses

Implied-goal recall (goals never stated, recoverable only by entailment):

| k | implied recall | depth-1 exact | depth-2 exact | depth-3 exact |
|---|---|---|---|---|
| 0 | 0.968 | 1.00 | 0.98 | 0.82 |
| 1 | 0.989 | 1.00 | 0.98 | 0.94 |
| 2 | **1.000** | 1.00 | 1.00 | 1.00 |
| 4 | 1.000 | 1.00 | 1.00 | 1.00 |
| 8 | 1.000 | 1.00 | 1.00 | 1.00 |

Two things worth noting:

1. **The effect is concentrated exactly where the design predicted.** Depth-1 is at
   ceiling for every k. Depth-3 carries the whole gradient: 0.82 -> 0.94 -> 1.00.
   Extra latent compute only mattered on items needing more entailment hops.
2. **The collapse diagnostic predicts the compute ceiling.** Performance saturates
   at k=2; the trajectory is 0.98-converged by k~3. The useful latent compute is
   the pre-collapse steps, and nothing after. If this holds up, `cos_steps` is a
   cheap way to predict where a k-sweep will flatten without running the sweep.

## What this does NOT yet show

- **The train/test k-mismatch confound is uncontrolled.** The model was trained only
  at k=4. Evaluating at k=0 and k=1 is off-distribution, so the low scores there may
  reflect distribution shift rather than insufficient compute. Until we train a
  separate model per k (or sample k during training), "more latent compute helped"
  is NOT supportable from this run.
- **The task is saturated.** Ceiling is 1.00 and it is reached by k=2, so k=2 vs k=8
  is indistinguishable. The instrument has no headroom.
- **n=50 per depth, one seed.** The depth-3 move 0.82 -> 1.00 is ~9 items. No CI.

## Next

1. Train with k sampled uniformly from {0,1,2,4,8} so every test-time k is in
   distribution. This is the control that decides whether Result 2 survives.
2. Harden the task until k=0 sits well below ceiling: longer chains, larger goal
   vocabulary, distractor phrases, goals requiring two seeds to co-occur.
