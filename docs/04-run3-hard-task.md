# Run 3: headroom restored, confound removed, curve is flat (2026-09-18)

Run 2 refuted run 1 but left the task at ceiling, so it could not detect a real
effect even if one existed. Run 3 fixes that.

## The task fix

In v1 the entailment graph was global and fixed, so 8.8M LoRA params memorised it.
In v2 **every example carries its own randomly generated rule set in the prompt**.
The graph cannot live in the weights; the model must read and traverse novel rules.
Verified: **0 of 400 val rule-sets appear anywhere in train**. Prompt grew 30 -> 73
tokens. Implied goals scale 1.46 / 2.92 / 4.56 across depth 1/2/3.

Training used the run-2 control from the start: k balanced over {0,1,2,4,8}, so no
test-time k is off-distribution. 750 steps, bs=6, 646s, peak 12.2 GB. Final loss
0.0682 -- **not** saturated, unlike run 2's 0.0002.

## The result: headroom exists and the curve is flat

| k | implied recall | d1 exact | d2 exact | d3 exact |
|---|---|---|---|---|
| 0 | 0.964 | 0.96 | 0.92 | **0.76** |
| 1 | 0.968 | 0.96 | 0.92 | 0.78 |
| 2 | 0.970 | 0.96 | 0.94 | 0.76 |
| 4 | 0.970 | 0.96 | 0.90 | 0.80 |
| 8 | 0.968 | 0.96 | 0.92 | 0.78 |

Depth-3 exact sits at 0.76 with k=0, so there is **24 points of headroom** for latent
compute to claim. It claims essentially none. Implied recall moves 0.964 -> 0.970 and
then back down at k=8; depth-3 exact wanders 0.76-0.80, a spread of 2 items out of 50,
non-monotonic in k.

Train loss points the same way, if anything against the hypothesis:
`{k=0: 0.048, k=1: 0.044, k=2: 0.075, k=4: 0.075, k=8: 0.080}`. More latent steps fit
the training data slightly *worse*.

## Reading this honestly

With one seed and n=50 per stratum we cannot resolve small effects; a 2-point move is
noise. What we *can* rule out is an effect anywhere near the size the confound
manufactured: run 1 showed 18 points at depth-3 (0.82 -> 1.00), and here the entire
observed spread is at most 4 points and does not increase with k.

So: on a task where memorisation is impossible, headroom is 24 points, and the
k-mismatch confound is controlled, **eight latent steps are worth about the same as
zero**.

The collapse diagnostic tightened once more: 0.954 -> 0.999, the tightest of the four
trainings (untrained 0.963->0.987, run1 0.929->0.993, run2 0.950->0.998). Across every
configuration tried, training makes the latent loop converge to its fixed point
*faster*, never slower. The loop is not learning to use the extra steps.

## Honest limits

- One seed. Multi-seed CIs would be needed to claim a *precise* null.
- One model size (0.5B) and one adapter budget. A larger model, or full fine-tuning
  rather than LoRA, might behave differently.
- One latent variant: last-hidden-state feedback (Coconut-style). Does not speak to
  looped-block or pause-token variants.
