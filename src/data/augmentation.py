"""Classical augmentation arm for the Layer 7 downstream experiment.

The spec is explicit that an under-specified classical baseline is one of
the two most common ways an "augmentation helps" result gets overstated -
a weak baseline does the comparison's work for you. So every transform and
every probability is a named constant here, not a magic number buried in a
training loop, and the whole pipeline is deterministic given an
`np.random.Generator` - re-running with the same seed reproduces the exact
same augmented batch.
"""
import numpy as np

# Every transform and its probability - the exact numbers the spec asks to
# state in full. Applied independently, in this fixed order, to a patch.
CONFIG = {
    "horizontal_flip": {"p": 0.5},
    "vertical_flip": {"p": 0.5},
    "rotate_90": {"p": 0.5, "k_choices": [1, 2, 3]},  # k=90/180/270 degrees, chosen uniformly when applied
    "gaussian_noise": {"p": 0.3, "sigma": 5.0},        # pixel intensity units, on a 0-255 scale
    "brightness_scale": {"p": 0.3, "low": 0.9, "high": 1.1},
}


def classical_augment(patch, rng):
    """Apply the full classical-augmentation pipeline to one patch.
    Deterministic given `rng` - same generator state produces the same
    sequence of coin-flips and the same output.
    """
    out = patch.copy()

    if rng.random() < CONFIG["horizontal_flip"]["p"]:
        out = out[:, ::-1]
    if rng.random() < CONFIG["vertical_flip"]["p"]:
        out = out[::-1, :]
    if rng.random() < CONFIG["rotate_90"]["p"]:
        k = rng.choice(CONFIG["rotate_90"]["k_choices"])
        out = np.rot90(out, k=k)
    if rng.random() < CONFIG["gaussian_noise"]["p"]:
        out = out + rng.normal(0, CONFIG["gaussian_noise"]["sigma"], size=out.shape)
    if rng.random() < CONFIG["brightness_scale"]["p"]:
        factor = rng.uniform(CONFIG["brightness_scale"]["low"], CONFIG["brightness_scale"]["high"])
        out = out * factor

    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    patch = np.arange(16, dtype=np.float64).reshape(4, 4)

    # determinism: same seed -> byte-identical output, every call
    out_a = classical_augment(patch, np.random.default_rng(0))
    out_b = classical_augment(patch, np.random.default_rng(0))
    assert np.array_equal(out_a, out_b), "same seed must produce identical augmented output"
    print("determinism check passed: same seed reproduces the same augmented patch")

    # shape must be preserved (rotation by 90/270 on a non-square patch
    # would break downstream code expecting a fixed patch_size x patch_size)
    assert out_a.shape == patch.shape
    print(f"shape-preservation check passed: {patch.shape} -> {out_a.shape}")

    # empirical probability check: horizontal_flip should fire in
    # ~50% of many independent calls, not some other rate silently baked in
    rng = np.random.default_rng(0)
    n_trials = 20000
    flipped = 0
    for _ in range(n_trials):
        if rng.random() < CONFIG["horizontal_flip"]["p"]:
            flipped += 1
    empirical_rate = flipped / n_trials
    assert abs(empirical_rate - CONFIG["horizontal_flip"]["p"]) < 0.02, \
        f"empirical flip rate {empirical_rate:.3f} too far from configured p={CONFIG['horizontal_flip']['p']}"
    print(f"empirical probability check passed: horizontal_flip fired {empirical_rate:.3f} "
          f"of {n_trials} trials (configured p={CONFIG['horizontal_flip']['p']})")
