"""Single-scale iterative sampler: turns pure noise into a synthetic image
by repeatedly denoising overlapping patches against a real patch bank, with
a shrinking noise level at each step.
"""
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.patches import extract_patches, gaussian_window, reconstruct_from_patches  # noqa: E402
from data.pyramid import detail_sigmas  # noqa: E402
from models.denoiser import denoise_patch_approx, denoise_patches_batch, select_patches_nn  # noqa: E402


def match_histogram(image, reference):
    """Exact histogram specification: return `image` remapped so its
    intensity histogram equals `reference`'s, preserving the RANK order of
    every pixel (so spatial structure is untouched - only the tone curve
    changes).

    WHY THIS EXISTS: this sampler's variance-preserving renormalization
    matches the real image's MEAN and STD - the first two moments only. A
    real electron-microscopy field of sparse particles on a bright support
    has a strongly ASYMMETRIC, bimodal histogram (measured on our own five
    min-dataset images: 18-39% of pixels are particle, skew -1.3 to -4.3).
    Two moments cannot encode that, so the sampler is free to satisfy them
    with a balanced ~50/50 texture instead, which is exactly the observed
    failure - particles merging into connected "continents" with far too
    little background between them (measured: 23% real particle area
    rendered as 57%). Constraining the full marginal histogram is the
    classical fix, going back to Heeger & Bergen 1995 ("Pyramid-Based
    Texture Analysis/Synthesis"), which matches histograms at every pyramid
    level for precisely this reason.

    This is per-image and fully data-driven - the reference is that image's
    own pyramid level, with no tuned constant - so it does not encode a
    preference for any particular image's appearance.
    """
    flat = np.asarray(image, dtype=np.float64).ravel()
    ref_sorted = np.sort(np.asarray(reference, dtype=np.float64).ravel())
    order = np.argsort(flat, kind="stable")
    # resample the reference's sorted values onto however many pixels the
    # target has, so the two do not need to be the same size
    positions = np.linspace(0, len(ref_sorted) - 1, len(flat))
    out = np.empty_like(flat)
    out[order] = np.interp(positions, np.arange(len(ref_sorted)), ref_sorted)
    return out.reshape(np.asarray(image).shape)


def estimate_sigma_range(patch_bank, num_samples=2000, seed=0):
    """Derive a sensible (sigma_max, sigma_min) from the bank's own
    patch-to-patch distance distribution, instead of a hardcoded constant
    that only happens to fit one particular image.
    """
    rng = np.random.default_rng(seed)
    flat = patch_bank.reshape(len(patch_bank), -1)
    idx_a = rng.integers(0, len(flat), num_samples)
    idx_b = rng.integers(0, len(flat), num_samples)
    distances = np.sqrt(np.sum((flat[idx_a] - flat[idx_b]) ** 2, axis=1))

    sigma_max = np.percentile(distances, 50)  # "typical" separation: permissive but not everything-looks-alike
    sigma_min = np.percentile(distances, 5)   # near the closest-neighbor scale: selective

    # A bank that is mostly one flat value (e.g. a synthetic mostly-black
    # control image, or a near-uniform coarsest pyramid level) can have so
    # many identical patches that even the MEDIAN distance is 0 - not just
    # the 5th percentile. Fall back to the bank's own pixel value range,
    # which is never 0 unless the whole bank is a single constant.
    if sigma_max <= 1e-6:
        value_range = patch_bank.max() - patch_bank.min()
        sigma_max = value_range if value_range > 1e-6 else 1.0

    # a small or near-uniform bank (e.g. the coarsest pyramid level) can have
    # many identical/near-identical patches, making the 5th percentile land
    # on exact duplicates - distance 0. A geometric sigma schedule cannot
    # include 0, and sigma=0 would divide by zero in denoise_patch, so floor
    # sigma_min to a small fraction of sigma_max instead of letting it hit 0.
    sigma_min = max(sigma_min, sigma_max * 0.01)
    return sigma_max, sigma_min


def sample_single_scale(shape, patch_bank, patch_size, stride, num_steps, sigma_max, sigma_min, seed, step_fraction=0.5, init=None, mean_tolerance=None, topk=None, window=None, ddim=False, eta=0.0, jitter=False, nn_alpha=None, robust_norm=None):
    """step_fraction controls how far each step moves toward the denoised
    mean (Langevin-style partial step) instead of jumping fully onto it -
    a full jump (step_fraction=1) collapses variance almost immediately,
    since averaging is inherently variance-reducing.

    init: start from this image instead of pure noise - used for
    refinement, where the starting point already has coarse structure
    from a previous scale and only needs detail added, not a fresh start.

    mean_tolerance: None runs the exact denoiser (denoise_patch) as before.
    A number switches every patch call to the approximate, mean-prefiltered
    denoiser (denoise_patch_approx) with that tolerance - lets a caller
    build a "fast" configuration without duplicating this whole function.

    topk: forwarded to the denoiser - restrict each patch's weighted
    average to its k nearest bank patches instead of the full bank.
    None (default) is the exact full-bank behavior, unchanged.

    window: forwarded to reconstruct_from_patches - a patch_size x
    patch_size weight matrix (e.g. gaussian_window) instead of uniform
    per-pixel averaging when patches are stitched back together each step.
    None (default) is the exact uniform-averaging behavior, unchanged.

    robust_norm: forwarded to reconstruct_from_patches - use the robust
    (IRLS, Kwatra et al. 2005) aggregation instead of the plain weighted
    mean when stitching patches back together. None (default) is unchanged.
    This is the SECOND of the two averaging stages in this pipeline: even
    with nn_alpha set (so every estimated patch is an exact, sharp real
    patch), stride < patch_size means each pixel is still the mean of
    several overlapping patches, which re-blurs any edge those patches
    disagree about. The two knobs address different stages and compose.

    ddim=False (default): the original Langevin-style update - a partial
    step toward the denoised mean, plus injected per-pixel noise, plus
    variance-preserving renormalization (see below) - byte-identical to
    this function before ddim existed.

    eta: only used when ddim=True. The original Qiu paper's DDIM step is
    not fully deterministic - it blends the deterministic residual carry
    with a bit of fresh noise at each step, controlled by eta in [0, 1]
    (eta=0 recovers the pure deterministic update; eta=1 is closest to the
    Langevin path's full noise injection). eta=0.0 (default) keeps this
    function's exact prior deterministic behavior - the L5 spec requires
    the single-scale baseline stay unchanged, so this only activates when
    a caller explicitly asks for it.

    ddim=True: a deterministic DDIM-style update instead -
    x = x_hat + sigma_next * ((x - x_hat) / sigma) - which steps toward the
    denoised estimate and rescales the leftover residual to the next
    noise level. That rescale is a ratio of two PATCH-space sigmas, so
    unlike the Langevin path it never needs a separate pixel-space
    conversion or injected noise. It DOES still need the same variance-
    preserving renormalization as the Langevin path, though: x_hat is
    itself a weighted patch average (inherently blurring, especially at
    the early high-sigma steps where the softmax spreads across many bank
    patches), and the deterministic residual carry cannot revive contrast
    that x_hat never had. A stage-by-stage std trace on a real image
    confirmed this - variance was already lost after the very first
    (coarsest-level) sketch, before any refinement ran - which is why
    renormalization is applied to both branches below, not skipped for
    ddim as originally assumed.

    jitter: False (default) pads x by exactly patch_size on every side, so
    the patch grid always starts at the same pixel offset - identical to
    this function's behavior before jitter existed. True instead redraws a
    random sub-stride offset (0..stride-1 in each axis) every step and pads
    asymmetrically so the grid starts there instead - the total padding per
    axis is unchanged (still 2*patch_size + (stride-1) once jitter_margin
    is folded in), so this only shifts WHERE the fixed grid sits, not how
    much of the image each step sees. Classical texture-synthesis quilting
    (Efros & Freeman 2001) hides patch seams by giving the overlap region
    room for a boundary cut or blend; our reconstruction already does a
    per-pixel weighted average (not a hard cut), so seams there come from
    adjacent, independently-matched patches disagreeing within a narrow
    overlap band, not from a bad window normalization (reconstruct_from_
    patches divides by the true local weight sum, so it cannot ring). A
    FIXED grid re-derives that same disagreement at the same pixels every
    single step, so any residual grid pattern compounds instead of
    averaging out across the sampler's many steps. Jittering the grid's
    phase each step is the same anti-aliasing-by-dithering principle used
    for stochastic sampling in rendering: any one step's seam now lands at
    a different pixel than the last, so across num_steps iterations the
    seam locations spread out and get smoothed by the per-pixel running
    renormalization instead of reinforcing one fixed lattice.

    nn_alpha: None (default) uses the softmax weighted-average denoiser -
    this function's behavior before this parameter existed, unchanged. A
    number switches patch estimation to HARD nearest-neighbour selection
    with GPNN's normalized distance (see denoiser.select_patches_nn), using
    that number as alpha_rel.

    Why this option exists: the weighted average is the correct DENOISER,
    but this project is synthesizing, not denoising. Averaging several real
    patches whose particle edges sit in slightly different places yields a
    patch with no sharp edge anywhere, and repeating that over many steps
    and overlapping windows is what dissolves real discrete particles into
    smooth blobs - the dominant measured failure on this project's images.
    With nn_alpha set, every estimated patch is an exact real patch, so no
    edge is ever averaged away at the source; overlapping patches are still
    blended by reconstruct_from_patches, but they are blended between real
    patches rather than between already-blurred means. Granot et al. 2022
    (CVPR, "Drop the GAN") make exactly this argument for the single-image
    generative setting and select by hard argmin throughout.

    Note that with nn_alpha set, `sigma` no longer affects WHICH patches are
    chosen (there is no softmax temperature any more) - it still drives the
    ddim/Langevin update and the noise schedule, so the coarse-to-fine
    annealing still works, but selection itself is scale-free.
    """
    assert stride <= patch_size, (
        f"stride ({stride}) must be <= patch_size ({patch_size}) - a larger "
        "stride leaves real gaps between patches that no pixel covers"
    )
    rng = np.random.default_rng(seed)
    sigmas = np.geomspace(sigma_max, sigma_min, num_steps)

    # sigma_max/sigma_min live in PATCH space (Euclidean distance over a
    # patch_size**2-dim vector - correct for the denoise_patch weighting
    # formula below, which compares in that same space). Per-PIXEL noise
    # injection is a different unit: reusing sigma_max directly here made
    # initial pixel values fall far outside the real image's range (e.g.
    # -1993..1900 for a 0..255 source), inflated by roughly sqrt(patch_size**2).
    # pixel_std - the bank's own per-pixel intensity spread - is the
    # dimensionally correct reference for pixel-space noise instead.
    pixel_std = patch_bank.std()

    x = init.copy() if init is not None else rng.normal(loc=128.0, scale=pixel_std, size=shape)
    history = [x.copy()]

    # Border pixels are covered by far fewer overlapping patches than
    # interior pixels (a corner gets 1 patch, an edge gets 2, the interior
    # gets 4 with patch_size=4/stride=2) - fewer patches averaged together
    # means less smoothing, so extreme (denoised) values cluster at the
    # border. Padding by reflection before each step gives every real
    # pixel the same interior-level coverage; cropping after reconstruction
    # removes the padding again, so the returned shape is unchanged.
    pad = patch_size
    jitter_margin = (stride - 1) if jitter else 0

    for step, sigma in enumerate(sigmas):
        if jitter_margin > 0:
            off_r = int(rng.integers(0, jitter_margin + 1))
            off_c = int(rng.integers(0, jitter_margin + 1))
        else:
            off_r = off_c = 0
        pad_top, pad_bottom = pad + off_r, pad + (jitter_margin - off_r)
        pad_left, pad_right = pad + off_c, pad + (jitter_margin - off_c)

        x_padded = np.pad(x, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect")
        record = extract_patches(x_padded, patch_size, stride)
        if nn_alpha is not None:
            # hard nearest-neighbour selection - every estimate is an exact
            # real patch, so edges are never averaged away (see docstring)
            denoised_patches, _ = select_patches_nn(record.patches, patch_bank, alpha_rel=nn_alpha)
        elif mean_tolerance is None:
            # batched over all patches in this step at once (GPU when
            # available) instead of a Python loop - see denoise_patches_batch
            denoised_patches, _ = denoise_patches_batch(record.patches, patch_bank, sigma, topk=topk)
        else:
            denoised_patches = np.stack([
                denoise_patch_approx(patch, patch_bank, sigma, mean_tolerance)[0] for patch in record.patches
            ])
        record.patches = denoised_patches
        denoised_padded = reconstruct_from_patches(record, x_padded.shape, window=window, robust_norm=robust_norm)
        denoised = denoised_padded[pad_top:pad_top + shape[0], pad_left:pad_left + shape[1]]
        is_last_step = step == len(sigmas) - 1

        if ddim:
            # deterministic step toward the denoised estimate, then rescale
            # the leftover residual (x - denoised) to the next noise level -
            # sigma_next/sigma is a ratio of two PATCH-space quantities, so
            # it needs no pixel_std conversion here.
            #
            # CORRECTED (measured, not assumed): this update alone still
            # erodes variance in practice, because `denoised` is itself a
            # weighted PATCH AVERAGE - inherently variance-reducing - and at
            # the early, high-sigma steps the softmax spreads weight across
            # many bank patches, so x_hat starts out already blurred well
            # below the bank's real contrast. The shrinking residual carried
            # forward by this formula cannot reintroduce contrast that x_hat
            # never had - it only preserves whatever variance survived into
            # x_hat, which is why the original "no renormalization needed"
            # claim here was wrong: a stage-by-stage std trace on a real
            # image showed the loss already present after the very first
            # (coarsest-level) sketch, before any refinement ran. Fixed by
            # applying the same variance-preserving renormalization used in
            # the Langevin branch below - it adds no randomness, so this
            # update is still fully deterministic given a seed.
            sigma_next = 0.0 if is_last_step else float(sigmas[step + 1])
            eps_hat = (x - denoised) / sigma
            if eta > 0.0 and not is_last_step:
                c = eta * sigma_next
                deterministic_part = np.sqrt(max(sigma_next ** 2 - c ** 2, 0.0))
                x = denoised + deterministic_part * eps_hat + c * rng.normal(size=shape)
            else:
                x = denoised + sigma_next * eps_hat
        else:
            # partial Langevin-style step toward the mean, plus noise scaled
            # to the CURRENT sigma (not the shrinking next_sigma) - this is
            # what keeps the process from collapsing to the mean image.
            # The injected noise stays in PIXEL units (pixel_std), scaled by
            # sigma/sigma_max - a unitless decay fraction that preserves the
            # original schedule's shape without reusing patch-space sigma
            # directly as a pixel-space standard deviation.
            x = x + step_fraction * (denoised - x)
            if not is_last_step:
                x = x + rng.normal(0, pixel_std * (sigma / sigma_max), size=shape)

        # Variance-preserving renormalization: repeated weighted-averaging
        # is inherently variance-reducing (Layer 4/5's documented "variance
        # erodes" finding), and neither correctly-scaled noise injection
        # (Langevin) nor a deterministic residual carry (DDIM) counteracts
        # that by itself over many steps - x drifts toward a flat image at
        # the bank's mean. Rescaling x's spread back to pixel_std after
        # every step keeps the process exploring the real contrast range
        # throughout, instead of only at the very first step. Applying this
        # to both branches keeps DDIM deterministic (no randomness added)
        # while fixing the same measured variance loss Langevin already
        # guards against.
        cur_std = x.std()
        if cur_std > 1e-6:
            x = x.mean() + (x - x.mean()) * (pixel_std / cur_std)

        history.append(x.copy())

    return x, history


def generate_coarse_sketch(pyramid, patch_size, stride, num_steps, seed, step_fraction=0.5, mean_tolerance=None, topk=None, window=None, ddim=False, sigma_override=None, eta=0.0, jitter=False, nn_alpha=None, robust_norm=None):
    """Run the single-scale sampler on only the coarsest pyramid level.

    At that tiny resolution, a patch covers a large fraction of the whole
    image, so "local" and "global" are nearly the same thing - which is
    exactly why the single-scale method can lay out large-scale structure
    here, even though it cannot at full resolution.

    patch_size/stride here are THIS call's own values, not necessarily the
    same ones used for the finer refinement levels - see
    sample_coarse_to_fine's coarse_patch_size/coarse_stride, which pass a
    larger patch here specifically because a small patch cannot see a whole
    particle at this tiny resolution (see that docstring for the evidence).

    sigma_override: (sigma_max, sigma_min) to use instead of deriving them
    from this level's own bank via estimate_sigma_range - used by
    sample_coarse_to_fine to hand each level its detail_sigmas-measured
    budget when ddim=True. None (default) keeps the original behavior.
    """
    coarsest_level = pyramid[-1]
    bank = extract_patches(coarsest_level, patch_size, stride).patches.astype(np.float64)
    if sigma_override is not None:
        sigma_max, sigma_min = sigma_override
    else:
        sigma_max, sigma_min = estimate_sigma_range(bank)

    sketch, history = sample_single_scale(
        shape=coarsest_level.shape,
        patch_bank=bank,
        patch_size=patch_size,
        stride=stride,
        num_steps=num_steps,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        seed=seed,
        step_fraction=step_fraction,
        mean_tolerance=mean_tolerance,
        topk=topk,
        window=window,
        ddim=ddim,
        eta=eta,
        jitter=jitter,
        nn_alpha=nn_alpha,
        robust_norm=robust_norm,
    )
    return sketch, history


def refine_at_scale(current, target_level, patch_size, stride, num_steps, seed, step_fraction=0.5, noise_scale=0.3, mean_tolerance=None, topk=None, window=None, ddim=False, sigma_override=None, eta=0.0, laplacian_blend=False, jitter=False, nn_alpha=None, robust_norm=None):
    """Upsample `current` (a coarser-scale result) to `target_level`'s
    resolution, then add detail by denoising against a bank built from the
    REAL image at that resolution - not by generating from scratch.

    sigma_override: (sigma_max, sigma_min) for this level, already scaled
    to this level's own noise budget - used when ddim=True, in place of
    estimate_sigma_range(bank) * noise_scale. None (default) keeps the
    original behavior.

    laplacian_blend: False (default) returns the sampler's raw multi-step
    result unchanged - every full denoising step re-derives the WHOLE
    image from the bank, so nothing stops a later step from drifting the
    coarse shape `upsampled` already got right (the Qiu paper's own
    coarse-to-fine composes scales instead of letting each one fully
    override the last). True composes the output as
    upsampled + (refined - blur(refined)) - the coarse structure always
    comes verbatim from the previous, already-validated scale, and this
    level only contributes its own high-frequency residual on top. The
    blur sigma reuses build_pyramid's own formula, (1/scale_factor)/2,
    with the local scale factor recovered from the two shapes involved -
    no extra parameter needed since it is already implied by the pyramid.
    """
    zoom_factors = (target_level.shape[0] / current.shape[0], target_level.shape[1] / current.shape[1])
    upsampled = zoom(current, zoom_factors, order=1)

    bank = extract_patches(target_level, patch_size, stride).patches.astype(np.float64)
    pixel_std = bank.std()
    rng = np.random.default_rng(seed)

    if sigma_override is not None:
        sigma_max, sigma_min = sigma_override
        # DDIM's own update rule (see sample_single_scale) already carries
        # forward exactly the residual its schedule calls for - it does not
        # need a separately blended noisy_init the way the Langevin path
        # does. A small amount of this level's own measured noise budget
        # still seeds room for new detail on top of the upsampled structure.
        noisy_init = upsampled + rng.normal(0, sigma_max, size=target_level.shape)
    else:
        sigma_max, sigma_min = estimate_sigma_range(bank)
        # only a little fresh noise - most of the structure should already be
        # right from the upsampled coarser result; this just gives the
        # refinement room to add detail rather than only smoothing the upsample.
        # Noise here is per-pixel, so it must be scaled by pixel_std (not
        # sigma_max, which lives in PATCH space - see the same pixel_std-vs-
        # sigma_max distinction documented in sample_single_scale below). Reusing
        # sigma_max directly here made this noise 2-5x too large, drowning out
        # the coarse structure the previous scale had already laid out.
        noisy_init = upsampled + rng.normal(0, pixel_std * noise_scale, size=target_level.shape)
        sigma_max = sigma_max * noise_scale

    refined, history = sample_single_scale(
        shape=target_level.shape,
        patch_bank=bank,
        patch_size=patch_size,
        stride=stride,
        num_steps=num_steps,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        seed=seed,
        step_fraction=step_fraction,
        init=noisy_init,
        mean_tolerance=mean_tolerance,
        topk=topk,
        window=window,
        ddim=ddim,
        eta=eta,
        jitter=jitter,
        nn_alpha=nn_alpha,
        robust_norm=robust_norm,
    )

    if not laplacian_blend:
        return refined, history

    local_scale_factor = current.shape[0] / target_level.shape[0]
    blur_sigma = (1.0 / local_scale_factor) / 2.0
    detail = refined - gaussian_filter(refined, sigma=blur_sigma)
    blended = upsampled + detail

    blended_std = blended.std()
    if blended_std > 1e-6:
        blended = blended.mean() + (blended - blended.mean()) * (pixel_std / blended_std)

    history = history[:-1] + [blended.copy()]
    return blended, history


def sample_coarse_to_fine(pyramid, patch_size, stride, num_steps_per_scale, seed, step_fraction=0.5, mean_tolerance=None, topk=None, window_sigma=None, ddim=False, sigma_max=None, floor_ratio=0.1, eta=0.0, laplacian_blend=False, jitter=False, coarse_patch_size=None, coarse_stride=None, nn_alpha=None, robust_norm=None, histogram_match=False):
    """Full coarse-to-fine generation: lay out global structure at the
    coarsest scale, then add detail one pyramid level at a time, using each
    level's own real patches - see generate_coarse_sketch and
    refine_at_scale for what happens at each stage.

    mean_tolerance: forwarded to every sample_single_scale call - None for
    the exact reference configuration, a number to build a "fast"
    configuration using the approximate denoiser at every stage.

    topk: forwarded to every sample_single_scale call - restrict the
    softmax weighting to the k nearest bank patches. None (default) is the
    original full-bank behavior.

    window_sigma: None (default) is the original uniform-averaging
    reconstruction. A number builds a gaussian_window(patch_size,
    window_sigma) used at every scale instead.

    ddim: False (default) is the original Langevin-style sampler at every
    scale, unchanged. True switches every scale to the deterministic DDIM
    update, with each level's noise budget taken from detail_sigmas (how
    much real detail that level measurably adds over an upsample of the
    next-coarser level) instead of estimate_sigma_range's generic
    bank-distance percentiles.

    sigma_max, floor_ratio: only used when ddim=True - sigma_max seeds
    detail_sigmas' coarsest-level budget (None derives it from that
    level's own pixel std); floor_ratio sets each level's sigma_min as a
    fraction of that level's sigma_max (mirroring estimate_sigma_range's
    own floor against a zero-distance/near-uniform level).

    eta: forwarded to every sample_single_scale call (ddim=True only) -
    0.0 (default) is the exact prior deterministic-DDIM behavior; see
    sample_single_scale for what a nonzero value does.

    laplacian_blend: forwarded to every refine_at_scale call - False
    (default) is the exact prior behavior (this level's full multi-step
    result used as-is). True composes each refined level as the previous
    level's upsampled structure plus only this level's own high-frequency
    residual, so a later scale's refinement cannot drift the shape an
    earlier, already-validated scale established - see refine_at_scale.
    Only applies to refinement steps, not the coarsest sketch (there is no
    coarser structure yet to preserve at that level).

    jitter: forwarded to every sample_single_scale call - False (default)
    is the exact prior fixed-grid behavior; True redraws the patch grid's
    phase every step at every scale, to spread any single-step patch-seam
    disagreement across different pixels instead of reinforcing it at the
    same fixed grid lines every step - see sample_single_scale.

    coarse_patch_size, coarse_stride: None (default) uses `patch_size`/
    `stride` for the coarsest-level sketch too, same as before either
    parameter existed. A number uses a DIFFERENT (normally larger) patch
    size/stride just for generate_coarse_sketch, while every refine_at_scale
    call still uses the original patch_size/stride unchanged.

    Why this is needed: at the coarsest pyramid level, a full-resolution-
    sized patch (small on purpose, so fine levels don't blur/copy-paste
    real detail) can be smaller than a single particle at that tiny
    resolution - e.g. a ~30px-wide coarsest level with a 7-8px particle
    and patch_size=4 only sees particle FRAGMENTS, never a whole particle,
    so it cannot match/reproduce discrete particle shapes or their spacing,
    only local blur. This is the classical texture-synthesis finding from
    Efros & Leung 1999 ("Texture Synthesis by Non-parametric Sampling"):
    a matching window smaller than the pattern's own regular structure
    loses that structure, producing a locally-plausible but globally
    incoherent result - exactly the "real circle lattice -> few soft
    blobs" failure measured on this project's own real images (see
    docs/findings.md). Because refine_at_scale intentionally only ADDS
    detail on top of the previous scale's upsampled result (so a later
    scale cannot drift an already-good coarse layout), a wrong coarse
    layout is never corrected downstream - it only gets sharper. Giving
    just the coarsest level a bigger patch (matched to the level's own
    particle scale) fixes the layout at its source, while leaving every
    finer level's patch_size untouched avoids re-introducing the
    blocking/stripe regression a uniformly-larger patch size caused at
    fine levels (see docs/findings.md's patch-grid-boundary-artifacts
    entry - that candidate was rejected precisely because it changed
    EVERY level's patch size at once, conflating this fix with that one).
    """
    window = None if window_sigma is None else gaussian_window(patch_size, window_sigma)
    budgets = detail_sigmas(pyramid, sigma_max=sigma_max) if ddim else None

    def override_for(level_index):
        if not ddim:
            return None
        budget = budgets[level_index]
        return (budget, max(budget * floor_ratio, 1e-6))

    sketch_patch_size = patch_size if coarse_patch_size is None else coarse_patch_size
    sketch_stride = stride if coarse_stride is None else coarse_stride
    sketch, _ = generate_coarse_sketch(
        pyramid, sketch_patch_size, sketch_stride, num_steps_per_scale, seed, step_fraction,
        mean_tolerance=mean_tolerance, topk=topk,
        window=(window if coarse_patch_size is None else (None if window_sigma is None else gaussian_window(sketch_patch_size, window_sigma))),
        ddim=ddim, sigma_override=override_for(len(pyramid) - 1), eta=eta, jitter=jitter,
        nn_alpha=nn_alpha, robust_norm=robust_norm,
    )
    if histogram_match:
        sketch = match_histogram(sketch, pyramid[-1])
    stages = [sketch]

    current = sketch
    for level_index in range(len(pyramid) - 2, -1, -1):
        # pyramid[-1] is coarsest (already used); walk back toward
        # pyramid[0], the full resolution, one level at a time
        target_level = pyramid[level_index]
        current, _ = refine_at_scale(
            current, target_level, patch_size, stride, num_steps_per_scale, seed, step_fraction,
            mean_tolerance=mean_tolerance, topk=topk, window=window, ddim=ddim,
            sigma_override=override_for(level_index), eta=eta, laplacian_blend=laplacian_blend, jitter=jitter,
            nn_alpha=nn_alpha, robust_norm=robust_norm,
        )
        if histogram_match:
            current = match_histogram(current, target_level)
        stages.append(current)

    return current, stages


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # a small synthetic "source image" with real repeating structure -
    # a checkerboard, so a working sampler should visibly recover blocky
    # structure instead of staying pure noise
    block = np.zeros((16, 16))
    block[::4, :] = 200.0
    block[:, ::4] = 200.0
    source_bank_record = extract_patches(block, patch_size=4, stride=2)
    bank = source_bank_record.patches.astype(np.float64)

    final_image, history = sample_single_scale(
        shape=(16, 16),
        patch_bank=bank,
        patch_size=4,
        stride=2,
        num_steps=10,
        sigma_max=50.0,
        sigma_min=1.0,
        seed=0,
    )

    print(f"history has {len(history)} snapshots (initial noise + {len(history)-1} steps)")
    print(f"initial std: {history[0].std():.2f}, final std: {final_image.std():.2f}")

    assert final_image.shape == (16, 16)
    assert not np.array_equal(history[0], final_image), "sampler must change the image, not return the input noise unchanged"
    print("sanity check passed: sampler transformed pure noise into a different image")

    # ddim=True must run without crashing, stay finite, and hold onto real
    # variance across the run. This step's own renormalization (shared with
    # the Langevin branch, added after a real-image trace showed the DDIM
    # update alone still loses variance - the denoiser's weighted average is
    # inherently blurring, and a deterministic residual carry cannot revive
    # contrast x_hat never had) is what keeps this from collapsing.
    ddim_final, ddim_history = sample_single_scale(
        shape=(16, 16), patch_bank=bank, patch_size=4, stride=2, num_steps=10,
        sigma_max=50.0, sigma_min=1.0, seed=0, ddim=True,
    )
    assert np.all(np.isfinite(ddim_final))
    assert ddim_final.std() > 5.0, f"ddim path lost too much variance: std={ddim_final.std():.2f}"
    print(f"ddim single-scale check passed: final std={ddim_final.std():.2f}")

    # topk must restrict weighting without crashing the pipeline, and its
    # output should differ from the full-bank run (otherwise it silently
    # was not applied)
    topk_final, _ = sample_single_scale(
        shape=(16, 16), patch_bank=bank, patch_size=4, stride=2, num_steps=10,
        sigma_max=50.0, sigma_min=1.0, seed=0, topk=3,
    )
    assert not np.allclose(topk_final, final_image), "topk=3 must actually change the result vs. the full-bank run"
    print("topk single-scale check passed: restricted weighting changed the output")

    # window must restrict/blend reconstruction without crashing, and
    # (same logic) should change the result vs. uniform averaging
    from data.patches import gaussian_window as _gaussian_window
    windowed_final, _ = sample_single_scale(
        shape=(16, 16), patch_bank=bank, patch_size=4, stride=2, num_steps=10,
        sigma_max=50.0, sigma_min=1.0, seed=0, window=_gaussian_window(4),
    )
    assert not np.allclose(windowed_final, final_image), "a gaussian window must actually change the result vs. uniform averaging"
    print("windowed single-scale check passed: center-weighted reconstruction changed the output")

    # full coarse-to-fine with ddim=True + topk + window together, on a
    # real (tiny) pyramid, exercising detail_sigmas end to end
    from data.pyramid import build_pyramid as _build_pyramid
    toy_pyramid = _build_pyramid(block, num_scales=2, scale_factor=0.5)
    ddim_result, ddim_stages = sample_coarse_to_fine(
        toy_pyramid, patch_size=4, stride=2, num_steps_per_scale=5, seed=0,
        topk=3, window_sigma=0.25, ddim=True,
    )
    assert ddim_result.shape == block.shape
    assert np.all(np.isfinite(ddim_result))
    assert len(ddim_stages) == len(toy_pyramid)
    print(f"ddim coarse-to-fine check passed: final shape={ddim_result.shape}, std={ddim_result.std():.2f}")

    # the original (ddim=False, topk=None, window=None) coarse-to-fine call
    # must still work exactly as it did before any of these params existed
    original_result, _ = sample_coarse_to_fine(toy_pyramid, patch_size=4, stride=2, num_steps_per_scale=5, seed=0)
    assert original_result.shape == block.shape
    print("backward-compatible coarse-to-fine (no new params) still passes")

    # eta=0.0 (default) must be byte-identical to the pre-eta deterministic
    # DDIM path - this is the L5-mandated single-scale baseline, so eta
    # must be strictly opt-in
    eta0_final, _ = sample_single_scale(
        shape=(16, 16), patch_bank=bank, patch_size=4, stride=2, num_steps=10,
        sigma_max=50.0, sigma_min=1.0, seed=0, ddim=True, eta=0.0,
    )
    assert np.array_equal(eta0_final, ddim_final), "eta=0.0 must reproduce the deterministic ddim path exactly"
    print("eta=0.0 backward-compatibility check passed: identical to deterministic ddim")

    # a nonzero eta must inject real stochasticity (two different seeds
    # diverge) while staying finite and holding onto variance, mirroring
    # the L4 investigate question "what changes across random seeds?"
    eta_seed0, _ = sample_single_scale(
        shape=(16, 16), patch_bank=bank, patch_size=4, stride=2, num_steps=10,
        sigma_max=50.0, sigma_min=1.0, seed=0, ddim=True, eta=0.5,
    )
    eta_seed1, _ = sample_single_scale(
        shape=(16, 16), patch_bank=bank, patch_size=4, stride=2, num_steps=10,
        sigma_max=50.0, sigma_min=1.0, seed=1, ddim=True, eta=0.5,
    )
    assert np.all(np.isfinite(eta_seed0)) and np.all(np.isfinite(eta_seed1))
    assert not np.allclose(eta_seed0, eta_seed1), "eta>0 must make different seeds diverge"
    assert eta_seed0.std() > 5.0 and eta_seed1.std() > 5.0
    print("stochastic eta check passed: different seeds diverge, variance held")

    # laplacian_blend=False (default) must be byte-identical to the prior
    # coarse-to-fine behavior
    no_blend_result, _ = sample_coarse_to_fine(
        toy_pyramid, patch_size=4, stride=2, num_steps_per_scale=5, seed=0,
        topk=3, window_sigma=0.25, ddim=True, laplacian_blend=False,
    )
    assert np.array_equal(no_blend_result, ddim_result), "laplacian_blend=False must reproduce the prior coarse-to-fine result exactly"
    print("laplacian_blend=False backward-compatibility check passed")

    # laplacian_blend=True must run end to end, stay finite, hold onto
    # variance, and actually change the result vs. the unblended run
    blend_result, blend_stages = sample_coarse_to_fine(
        toy_pyramid, patch_size=4, stride=2, num_steps_per_scale=5, seed=0,
        topk=3, window_sigma=0.25, ddim=True, laplacian_blend=True,
    )
    assert blend_result.shape == block.shape
    assert np.all(np.isfinite(blend_result))
    assert len(blend_stages) == len(toy_pyramid)
    assert not np.allclose(blend_result, ddim_result), "laplacian_blend=True must actually change the result"
    print(f"laplacian_blend=True check passed: final shape={blend_result.shape}, std={blend_result.std():.2f}")

    # jitter=False (default) must be byte-identical to the prior fixed-grid
    # single-scale behavior
    no_jitter_final, _ = sample_single_scale(
        shape=(16, 16), patch_bank=bank, patch_size=4, stride=2, num_steps=10,
        sigma_max=50.0, sigma_min=1.0, seed=0, ddim=True, jitter=False,
    )
    assert np.array_equal(no_jitter_final, ddim_final), "jitter=False must reproduce the fixed-grid ddim path exactly"
    print("jitter=False backward-compatibility check passed")

    # jitter=True must run without crashing, stay finite, hold onto real
    # variance, and actually change the result vs. the fixed-grid run (same
    # style of check as eta/laplacian_blend above - a deeper "does this
    # reduce the grid artifact" judgment is made from the real-image runs,
    # not this toy self-test)
    jitter_final, _ = sample_single_scale(
        shape=(16, 16), patch_bank=bank, patch_size=4, stride=2, num_steps=10,
        sigma_max=50.0, sigma_min=1.0, seed=0, ddim=True, jitter=True,
    )
    assert np.all(np.isfinite(jitter_final))
    assert jitter_final.std() > 5.0, f"jitter path lost too much variance: std={jitter_final.std():.2f}"
    assert not np.allclose(jitter_final, no_jitter_final), "jitter=True must actually change the result vs. the fixed grid"
    print(f"jitter=True single-scale check passed: final shape={jitter_final.shape}, std={jitter_final.std():.2f}")

    # jitter must also work end to end through the full coarse-to-fine path
    jitter_c2f_result, jitter_c2f_stages = sample_coarse_to_fine(
        toy_pyramid, patch_size=4, stride=2, num_steps_per_scale=5, seed=0,
        topk=3, window_sigma=0.25, ddim=True, jitter=True,
    )
    assert jitter_c2f_result.shape == block.shape
    assert np.all(np.isfinite(jitter_c2f_result))
    assert len(jitter_c2f_stages) == len(toy_pyramid)
    assert not np.allclose(jitter_c2f_result, ddim_result), "jitter=True must actually change the coarse-to-fine result"
    print(f"jitter=True coarse-to-fine check passed: final shape={jitter_c2f_result.shape}, std={jitter_c2f_result.std():.2f}")

    # coarse_patch_size=None (default) must be byte-identical to the prior
    # coarse-to-fine behavior (uniform patch_size at every level)
    no_coarse_override_result, _ = sample_coarse_to_fine(
        toy_pyramid, patch_size=4, stride=2, num_steps_per_scale=5, seed=0,
        topk=3, window_sigma=0.25, ddim=True, coarse_patch_size=None, coarse_stride=None,
    )
    assert np.array_equal(no_coarse_override_result, ddim_result), "coarse_patch_size=None must reproduce the uniform-patch-size result exactly"
    print("coarse_patch_size=None backward-compatibility check passed")

    # coarse_patch_size set must run end to end (bigger patch than the toy
    # pyramid's own coarsest level still has to fit), stay finite, hold onto
    # variance, and actually change the result vs. the uniform-patch-size run
    small_toy_pyramid = _build_pyramid(block, num_scales=2, scale_factor=0.5)  # coarsest level is 8x8
    coarse_override_result, coarse_override_stages = sample_coarse_to_fine(
        small_toy_pyramid, patch_size=4, stride=2, num_steps_per_scale=5, seed=0,
        topk=3, window_sigma=0.25, ddim=True, coarse_patch_size=6, coarse_stride=3,
    )
    baseline_small_result, _ = sample_coarse_to_fine(
        small_toy_pyramid, patch_size=4, stride=2, num_steps_per_scale=5, seed=0,
        topk=3, window_sigma=0.25, ddim=True,
    )
    assert coarse_override_result.shape == block.shape
    assert np.all(np.isfinite(coarse_override_result))
    assert len(coarse_override_stages) == len(small_toy_pyramid)
    assert not np.allclose(coarse_override_result, baseline_small_result), "coarse_patch_size must actually change the result vs. uniform patch_size"
    print(f"coarse_patch_size=6 check passed: final shape={coarse_override_result.shape}, std={coarse_override_result.std():.2f}")

    # robust_norm=None (default) must be byte-identical to the prior behavior
    no_robust_result, _ = sample_coarse_to_fine(
        toy_pyramid, patch_size=4, stride=2, num_steps_per_scale=5, seed=0,
        topk=3, window_sigma=0.25, ddim=True, robust_norm=None,
    )
    assert np.array_equal(no_robust_result, ddim_result), "robust_norm=None must reproduce the prior result exactly"
    print("robust_norm=None backward-compatibility check passed")

    # robust_norm set must run end to end, stay finite, actually change the
    # result, and not collapse variance across the chained refinement stages
    # (the failure mode this pipeline is historically prone to - see the
    # variance-erosion findings in docs/findings.md).
    #
    # Deliberately NOT asserted here: that the output is "sharper". This toy
    # is a near-binary block pattern, where mean gradient magnitude measures
    # edge DENSITY, not edge sharpness - a mode-seeking aggregation that
    # cleans up spurious edges scores LOWER on it while being better, so the
    # assertion would encode a guess rather than the mechanism. The
    # sharpening mechanism is asserted directly and measurably where it is
    # well-defined instead: see the contested-edge check in data/patches.py.
    robust_result, _ = sample_coarse_to_fine(
        toy_pyramid, patch_size=4, stride=2, num_steps_per_scale=5, seed=0,
        topk=3, window_sigma=0.25, ddim=True, robust_norm=0.8,
    )
    assert robust_result.shape == ddim_result.shape
    assert np.all(np.isfinite(robust_result))
    assert not np.allclose(robust_result, ddim_result), "robust_norm must actually change the result"
    assert robust_result.std() > 0.5 * ddim_result.std(), "robust aggregation must not collapse variance"
    print(f"robust_norm=0.8 check passed: std {ddim_result.std():.2f} -> {robust_result.std():.2f}")

    # match_histogram, in isolation: the output's sorted values must equal
    # the reference's, and every pixel must keep its rank (structure intact)
    skewed = np.concatenate([np.full(90, 10.0), np.full(10, 200.0)]).reshape(10, 10)  # 10% "particles"
    balanced = rng.normal(100, 30, size=(10, 10))
    matched = match_histogram(balanced, skewed)
    assert np.allclose(np.sort(matched.ravel()), np.sort(skewed.ravel())), "matched image must carry the reference's exact histogram"
    # rank preservation, stated so it holds when the reference has TIES (as a
    # real bimodal histogram does): reading the output in the input's
    # ascending order must be non-decreasing, i.e. the remap is monotone.
    assert np.all(np.diff(matched.ravel()[np.argsort(balanced.ravel(), kind="stable")]) >= 0), \
        "histogram matching must be a monotone remap (structure preserved, only tone changed)"
    assert abs((matched > 100).mean() - 0.10) < 1e-9, \
        "a 10%-bright reference must yield a 10%-bright result, not the balanced input's ~50%"
    print("match_histogram check passed: exact reference histogram, rank order preserved")

    # histogram_match=False (default) must be byte-identical to prior behavior
    no_hist_result, _ = sample_coarse_to_fine(
        toy_pyramid, patch_size=4, stride=2, num_steps_per_scale=5, seed=0,
        topk=3, window_sigma=0.25, ddim=True, histogram_match=False,
    )
    assert np.array_equal(no_hist_result, ddim_result), "histogram_match=False must reproduce the prior result exactly"
    print("histogram_match=False backward-compatibility check passed")

    # histogram_match=True must make the OUTPUT's histogram match the real
    # full-resolution level's - which is the whole point of the parameter
    hist_result, _ = sample_coarse_to_fine(
        toy_pyramid, patch_size=4, stride=2, num_steps_per_scale=5, seed=0,
        topk=3, window_sigma=0.25, ddim=True, histogram_match=True,
    )
    assert np.allclose(np.sort(hist_result.ravel()), np.sort(toy_pyramid[0].ravel())), \
        "histogram_match=True must give the output the real level's exact histogram"
    print(f"histogram_match=True check passed: output histogram equals the real level's")
