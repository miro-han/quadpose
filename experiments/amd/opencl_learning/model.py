"""Analytic float32 CPU reference and matching AMD kernels."""

import ctypes as ct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .native import P


WIDTH = 8


@dataclass(frozen=True)
class Batch:
    cameras: np.ndarray
    hypotheses: np.ndarray
    valid: np.ndarray
    relation: np.ndarray
    labels: np.ndarray

    def first(self, count):
        return Batch(*(array[:count] for array in (self.cameras, self.hypotheses, self.valid, self.relation, self.labels)))


@dataclass(frozen=True)
class Model:
    camera_features: int
    hypothesis_features: int
    attention: bool

    @property
    def camera_end(self):
        return self.camera_features * WIDTH

    @property
    def readout_start(self):
        return self.camera_end + self.attention * self.hypothesis_features * WIDTH

    @property
    def bias(self):
        return self.readout_start + 2 * WIDTH

    @property
    def count(self):
        return self.bias + 1 + self.attention

    def initial(self, seed):
        rng = np.random.default_rng(seed)
        values = rng.normal(0, 0.14, self.count).astype(np.float32)
        values[self.bias] = 0
        if self.attention:
            values[self.bias + 1] = 0.2
        return values


def cpu_forward_backward(model, batch, theta, weights):
    x, k, y = batch.cameras, batch.hypotheses, batch.labels
    h = np.tanh(x @ theta[:model.camera_end].reshape(-1, WIDTH))
    local_weights, context_weights = theta[model.readout_start:model.bias].reshape(2, WIDTH)
    if model.attention:
        z = np.tanh(k @ theta[model.camera_end:model.readout_start].reshape(-1, WIDTH))
        scaling = np.float32(WIDTH ** -0.5)
        score = (h @ z.transpose(0, 2, 1)) * scaling + theta[model.bias + 1] * batch.relation
        valid = batch.valid[:, None] > 0
        score = np.where(valid, score, np.float32(-1e20))
        alpha = np.where(valid, np.exp(score - score.max(axis=2, keepdims=True)), 0)
        alpha /= np.maximum(alpha.sum(axis=2, keepdims=True), np.float32(1e-8))
        context = alpha @ z
    else:
        context = np.broadcast_to(h.mean(axis=1, keepdims=True), h.shape)
    logit = h @ local_weights + context @ context_weights + theta[model.bias]
    probability = 0.5 + 0.5 * np.tanh(0.5 * logit)
    weight = np.where(y > 0, weights[0], weights[1]).astype(np.float32)
    loss = np.mean((np.maximum(logit, 0) - logit * y + np.log1p(np.exp(-np.abs(logit)))) * weight)
    error = (probability - y) * weight / y.size
    dlocal = error[:, :, None] * local_weights
    dcontext = error[:, :, None] * context_weights
    gradient = np.empty_like(theta)
    if model.attention:
        dscore = alpha * (dcontext @ z.transpose(0, 2, 1) - np.sum(dcontext * context, axis=2, keepdims=True))
        dlocal += (dscore @ z) * scaling
        dz = alpha.transpose(0, 2, 1) @ dcontext + (dscore.transpose(0, 2, 1) @ h) * scaling
        gradient[model.camera_end:model.readout_start] = (k.reshape(-1, model.hypothesis_features).T @
            (dz * (1 - z**2)).reshape(-1, WIDTH)).ravel()
        gradient[model.bias + 1] = (dscore * batch.relation).sum()
    else:
        dlocal += dcontext.mean(axis=1, keepdims=True)
    gradient[:model.camera_end] = (x.reshape(-1, model.camera_features).T @ (dlocal * (1 - h**2)).reshape(-1, WIDTH)).ravel()
    gradient[model.readout_start:model.bias] = np.r_[np.sum(error[:, :, None] * h, axis=(0, 1)),
                                                    np.sum(error[:, :, None] * context, axis=(0, 1))]
    gradient[model.bias] = error.sum()
    return float(loss), probability, gradient


class AMDBatch:
    def __init__(self, amd, model, batch, theta, weights, program):
        self.amd, self.model = amd, model
        self.batch_size = len(batch.labels)
        self.weights = weights
        self.parameters = amd.buffer(theta)
        inputs = [amd.buffer(array) for array in (batch.cameras, batch.hypotheses, batch.valid, batch.relation, batch.labels)]
        self.gradients = amd.buffer(np.zeros((self.batch_size, model.count), dtype=np.float32))
        self.loss = amd.buffer(np.zeros(self.batch_size, dtype=np.float32))
        self.predictions = amd.buffer(np.zeros((self.batch_size, 4), dtype=np.float32))
        self.forward = amd.kernel(program, "forward_backward")
        self.update = amd.kernel(program, "sgd")
        amd.set_args(self.forward, *(P(handle) for handle in (*inputs, self.parameters, self.gradients, self.loss, self.predictions)),
                     *(ct.c_float(weight) for weight in weights))
        amd.set_args(self.update, P(self.parameters), P(self.gradients), ct.c_int(self.batch_size), ct.c_float(0.0))

    @staticmethod
    def program(amd, model):
        source = Path(__file__).with_name("model.cl").read_bytes()
        return amd.program(source, f"-cl-std=CL1.2 -DC={model.camera_features} -DK={model.hypothesis_features} "
                           f"-DH={WIDTH} -DATTENTION={int(model.attention)}")

    def reset(self, theta):
        self.amd.write(self.parameters, theta)

    def step(self, rate):
        self.amd.dispatch(self.forward, self.batch_size)
        learning_rate = ct.c_float(rate)
        self.amd.set_args(self.update, P(self.parameters), P(self.gradients), ct.c_int(self.batch_size), learning_rate)
        self.amd.dispatch(self.update, self.model.count)
        self.amd.finish()

    def read_parameters(self):
        return self.amd.read(self.parameters, np.empty(self.model.count, dtype=np.float32))

    def evaluate(self, with_gradient=False):
        self.amd.dispatch(self.forward, self.batch_size)
        loss = self.amd.read(self.loss, np.empty(self.batch_size, dtype=np.float32)).mean()
        probabilities = self.amd.read(self.predictions, np.empty((self.batch_size, 4), dtype=np.float32))
        gradient = None
        if with_gradient:
            gradient = self.amd.read(self.gradients, np.empty((self.batch_size, self.model.count), dtype=np.float32)).mean(axis=0)
        return float(loss), probabilities, gradient
