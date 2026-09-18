# Run 2: the control. Run 1's effect was an artifact. (2026-09-18)

Identical task, data, model, LoRA config and total examples seen (4800). The only
change: k is sampled per step over {0,1,2,4,8} instead of fixed at 4, so no
test-time k is off-distribution. 600 steps at bs=8, 514s on one T4.

## The result

Depth-3 exact match, the stratum that carried the entire run-1 effect:

| k | run 1 (trained k=4 only) | run 2 (k in-distribution) |
|---|---|---|
| 0 | 0.82 | **1.00** |
| 1 | 0.94 | **1.00** |
| 2 | 1.00 | 1.00 |
| 4 | 1.00 | 1.00 |
| 8 | 1.00 | 1.00 |

The gradient vanishes completely. Implied-goal recall is 1.000 at every k, in every
depth stratum. Training loss is flat across k too: k=0 reaches 0.0002 and k=8 reaches
0.0002.

**Verdict: run 1's apparent "more latent compute produces better answers" was
train/test k mismatch, not compute.** Zero latent steps is exactly as good as eight
once zero is in-distribution.

The collapse diagnostic tightened again (0.950 -> 0.998, vs 0.929 -> 0.993 in run 1
and 0.963 -> 0.987 untrained). Three trainings, three tighter fixed points. Nothing
we have done makes the latent loop keep doing new work.

## What this does and does not establish

It **does** establish that on this task the run-1 effect was an artifact, and that the
train/test k-mismatch confound is large enough to manufacture a clean-looking
compute-scaling curve (0.82 -> 1.00) out of nothing. Any latent-reasoning result that
trains at one k and sweeps k at test time should be treated as suspect until this
control is run.

It does **not** establish that latent compute never helps. The task is at ceiling:
1.00 everywhere. An instrument pinned at its maximum cannot detect a benefit that
exists. The hypothesis is untested on a task with headroom, not refuted in general.

## Next

Rebuild the task with real headroom so k=0 sits well below ceiling, then re-run this
same control. Until k=0 is beatable, the sweep cannot say anything.
