# Mechanism check (2026-09-18)

Ran on 2x Tesla T4 (15.6 GB each), torch 2.10.0+cu128, transformers 5.0.0,
Qwen2.5-0.5B in fp16.

## What was verified

| # | Check | Result |
|---|-------|--------|
| a | forward from `inputs_embeds` | OK, `hidden_states[-1]` = (1,12,896), 25 layers |
| b-d | 8 latent steps feeding hidden state back, no token sampled | OK, cache type `DynamicCache`, len 20 |
| e | gradients through 3 unrolled latent steps | OK, grad norm 2.93e+02 |
| - | peak GPU memory, 0.5B | 1.88 GB |

The mechanism survives the transformers 5 major version. Plenty of headroom for 1.5B.

## The finding that matters

On an **untrained** model the latent trajectory collapses to a fixed point:

```
cos(z_k, z_k+1):  +0.963 +0.966 +0.979 +0.985 +0.987 +0.988 +0.987
||z_k||:           311.0  305.9  310.2  319.3  326.9  330.9  332.6  332.8
```

Cosine similarity between consecutive latent states climbs toward 1.0 and the norm
plateaus around 332. After roughly 4-5 steps the loop is re-computing nearly the
same vector. Extra latent compute buys almost nothing.

This is the **floor of the mechanism**, measured before any training. It matters for
two reasons:

1. Any benefit of larger k must come from *training the model to use the steps*, not
   from the architecture alone. A k-sweep on an untrained or lightly-tuned model will
   produce a flat curve for trivial reasons.
2. It gives a free diagnostic to run throughout: if `cos_steps` climbs past ~0.99
   after training, the latent loop has collapsed again and the k-knob is inert. This
   is implemented as `LatentTrace.collapsed` in `src/intentions/latent.py`.

Not yet established: whether the collapse persists after training, and whether fp16
contributes (norm ~332 is far from the fp16 max of 65504, so overflow is not the cause).
