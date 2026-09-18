"""Coconut-style latent reasoning loop, verified on transformers 5.0.0 / T4 fp16.

The model thinks for k steps in continuous space before emitting anything: the last
hidden state is fed back as the next input embedding, so no token is ever sampled
during the thinking phase. k is the compute knob the experiment sweeps.
"""
from dataclasses import dataclass
import torch


@dataclass
class LatentTrace:
    """Diagnostics for one latent rollout. cos_steps is the health check: if it
    climbs toward 1.0 the trajectory has collapsed to a fixed point and extra
    latent steps are buying nothing."""
    cos_steps: list
    norms: list

    @property
    def collapsed(self, thresh: float = 0.99) -> bool:
        return bool(self.cos_steps) and self.cos_steps[-1] > thresh


def latent_forward(model, inputs_embeds, k: int, trace: bool = False):
    """Run k latent steps. Returns (last_hidden, cache, LatentTrace|None).

    k=0 is the matched control: identical params, identical weights, no extra
    thinking. Any effect of k must be measured against it.
    """
    out = model(inputs_embeds=inputs_embeds, output_hidden_states=True, use_cache=True)
    cache = out.past_key_values
    z = out.hidden_states[-1][:, -1:, :]

    cos, norms = [], [z.float().norm().item()]
    prev = z
    for _ in range(k):
        out = model(inputs_embeds=z, past_key_values=cache,
                    output_hidden_states=True, use_cache=True)
        cache = out.past_key_values
        z = out.hidden_states[-1][:, -1:, :]
        if trace:
            c = torch.nn.functional.cosine_similarity(
                prev.float().flatten(), z.float().flatten(), dim=0).item()
            cos.append(c)
            norms.append(z.float().norm().item())
        prev = z
    return z, cache, (LatentTrace(cos, norms) if trace else None)
