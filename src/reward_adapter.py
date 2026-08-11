"""Adapter that bridges single-example reward functions to TRL's batched API.

Each reward function scores one ``(prompt, completion, example)`` triple. TRL calls
rewards once per batch as ``f(prompts, completions, **columns)`` where the dataset
columns arrive as lists aligned to ``completions`` and the function must return a list
of floats. TRL also injects non-column keyword arguments (e.g. ``trainer_state``), so
we keep only the kwargs that are true per-completion lists before indexing.
"""
from reward_functions import (
    guess_value,
    output_format_check,
    uses_previous_feedback,
)


def make_trl_reward(single_fn):
    """Wrap f(prompt, completion, example) -> float into a batched TRL reward."""
    def trl_fn(prompts, completions, **kwargs):
        n = len(completions)
        # Keep only real per-completion columns; drop extras like trainer_state.
        cols = {k: v for k, v in kwargs.items()
                if isinstance(v, (list, tuple)) and len(v) == n}
        rewards = []
        for i in range(n):
            p, c = prompts[i], completions[i]
            if isinstance(p, list):   # conversational prompt -> last turn's text
                p = p[-1]["content"]
            if isinstance(c, list):   # conversational completion -> assistant text
                c = c[0]["content"]
            example = {k: v[i] for k, v in cols.items()}
            try:
                rewards.append(float(single_fn(p, c, example)))
            except Exception:
                rewards.append(0.0)
        return rewards

    trl_fn.__name__ = single_fn.__name__   # nice names in the training logs
    return trl_fn


def build_reward_funcs():
    """Return the three adapted reward functions used by GRPOTrainer."""
    return [
        make_trl_reward(output_format_check),
        make_trl_reward(uses_previous_feedback),
        make_trl_reward(guess_value),
    ]
