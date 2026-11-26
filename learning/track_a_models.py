"""Small deterministic NumPy learners for supervised geometry experiments."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Model:
    center: np.ndarray
    scale: np.ndarray
    weights: tuple[np.ndarray, ...]

    def normalized(self, x):
        return np.clip((x - self.center) / self.scale, -8, 8)

    def forward(self, x):
        return forward(self.normalized(x), self.weights)[0]

    def probability(self, x):
        return sigmoid(self.forward(x))


def sigmoid(x):
    return 0.5 + 0.5 * np.tanh(x * 0.5)


def forward(x, weights):
    if len(weights) == 2:
        return x @ weights[0] + weights[1][0], None
    hidden = np.tanh(x @ weights[0] + weights[1])
    return hidden @ weights[2] + weights[3][0], hidden


def gradient(x, hidden, weights, derivative):
    if hidden is None:
        return (x.T @ derivative, np.array([derivative.sum()], dtype=np.float32))
    hidden_gradient = derivative[:, None] * weights[2] * (1 - hidden**2)
    return (x.T @ hidden_gradient, hidden_gradient.sum(axis=0), hidden.T @ derivative,
            np.array([derivative.sum()], dtype=np.float32))


def fit(x, y, *, hidden_width=0, paired=None, seed=410, steps=450):
    """Fit classification logits or a shared pairwise ranking utility on train arrays only."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    paired = None if paired is None else np.asarray(paired, dtype=np.float32)
    if len(x) == 0 or len(y) != len(x) or (paired is not None and paired.shape != x.shape):
        raise ValueError("Expected nonempty, aligned training arrays")
    pool = x if paired is None else np.concatenate((x, paired))
    center, scale = pool.mean(axis=0), np.maximum(pool.std(axis=0), 0.025)
    rng = np.random.default_rng(seed)
    inputs = x.shape[1]
    if hidden_width:
        weights = (rng.normal(0, 0.5 / np.sqrt(inputs), (inputs, hidden_width)).astype(np.float32),
                   np.zeros(hidden_width, dtype=np.float32),
                   rng.normal(0, 0.25 / np.sqrt(hidden_width), hidden_width).astype(np.float32),
                   np.zeros(1, dtype=np.float32))
    else:
        weights = np.zeros(inputs, dtype=np.float32), np.zeros(1, dtype=np.float32)
    model = Model(center, scale, weights)
    normalized = model.normalized(x)
    other = None if paired is None else model.normalized(paired)
    positives = float(y.mean())
    prevalence = np.where(y > 0.5, 0.5 / max(positives, 1e-5), 0.5 / max(1 - positives, 1e-5)).astype(np.float32)
    moment = [np.zeros_like(value) for value in weights]
    variance = [np.zeros_like(value) for value in weights]
    for step in range(1, steps + 1):
        indices = rng.integers(len(x), size=min(384, len(x)))
        batch = normalized[indices]
        scores, hidden = forward(batch, weights)
        if other is not None:
            right_scores, right_hidden = forward(other[indices], weights)
            scores -= right_scores
        derivative = (sigmoid(scores) - y[indices]) * prevalence[indices] / len(indices)
        changes = gradient(batch, hidden, weights, derivative)
        if other is not None:
            right_changes = gradient(other[indices], right_hidden, weights, -derivative)
            changes = tuple(first + second for first, second in zip(changes, right_changes, strict=True))
        for value, change, average, squared in zip(weights, changes, moment, variance, strict=True):
            change += 0.0005 * value
            average *= 0.9
            average += 0.1 * change
            squared *= 0.999
            squared += 0.001 * change**2
            value -= 0.012 * (average / (1 - 0.9**step)) / (np.sqrt(squared / (1 - 0.999**step)) + 1e-7)
    return model
