"""Minimal single-hidden-layer MLP with plain SGD - the Layer 7 downstream
task's trainable model (binary: does this patch contain particle pixels?).

Implemented from scratch with numpy, not a framework, so "equal optimizer
update budget across every downstream arm" (spec rule 14) is a literal,
auditable step counter passed into a loop - not something hidden inside a
library's .fit(epochs=...) call where "one epoch" could silently mean a
different number of gradient updates depending on dataset size per arm.
"""
import numpy as np


class PatchClassifier:
    def __init__(self, input_dim, hidden_dim=16, seed=0):
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = rng.normal(0, scale1, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, scale2, size=(hidden_dim, 1))
        self.b2 = np.zeros(1)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = a1 @ self.W2 + self.b2
        p = 1.0 / (1.0 + np.exp(-np.clip(z2, -30, 30)))  # sigmoid, clipped to avoid overflow
        return p.ravel(), (X, z1, a1)

    def train_step(self, X, y, lr):
        """One SGD update on one batch. Returns the batch's BCE loss before
        the update - the caller counts calls to this, not "epochs", so the
        step count is the literal, comparable unit of training budget.
        """
        p, (X_, z1, a1) = self.forward(X)
        n = len(y)
        dz2 = (p - y).reshape(-1, 1) / n  # d(BCE)/dz2 for a sigmoid output
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)  # ReLU gradient
        dW1 = X_.T @ dz1
        db1 = dz1.sum(axis=0)

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

        loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
        return float(loss)

    def predict(self, X):
        p, _ = self.forward(X)
        return (p >= 0.5).astype(int)


def train(model, X_pool, y_pool, total_steps, batch_size, lr, seed):
    """Train for EXACTLY total_steps gradient updates, sampling batches
    with replacement from (X_pool, y_pool) each step - so total_steps and
    batch_size are the only things that determine how much training
    happens, independent of how large X_pool itself is. This is what makes
    "equal update budget across arms" possible even when the three arms'
    pools are different sizes (real-only vs real+augmented vs
    real+synthetic).
    """
    rng = np.random.default_rng(seed)
    n = len(y_pool)
    for _ in range(total_steps):
        idx = rng.integers(0, n, size=batch_size)
        model.train_step(X_pool[idx], y_pool[idx], lr)
    return model


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # a tiny linearly-separable toy dataset: label = 1 if sum of features > 0
    n_samples, input_dim = 400, 4
    X = rng.normal(0, 1, size=(n_samples, input_dim))
    y = (X.sum(axis=1) > 0).astype(float)

    model = PatchClassifier(input_dim=input_dim, hidden_dim=8, seed=0)
    initial_acc = (model.predict(X).ravel() == y).mean()

    model = train(model, X, y, total_steps=300, batch_size=32, lr=0.1, seed=1)
    final_acc = (model.predict(X).ravel() == y).mean()

    assert final_acc > initial_acc, f"training should improve accuracy: {initial_acc:.2f} -> {final_acc:.2f}"
    assert final_acc > 0.85, f"a trivially separable task should reach high accuracy, got {final_acc:.2f}"
    print(f"training check passed: accuracy {initial_acc:.2f} -> {final_acc:.2f} after 300 steps")

    # determinism: same seed (model init AND training) reproduces the same result
    model_a = train(PatchClassifier(input_dim, 8, seed=0), X, y, 300, 32, 0.1, seed=1)
    model_b = train(PatchClassifier(input_dim, 8, seed=0), X, y, 300, 32, 0.1, seed=1)
    assert np.array_equal(model_a.W1, model_b.W1), "same seeds must reproduce identical trained weights"
    print("determinism check passed: identical seeds reproduce identical trained weights")
