"""Run the entire AMAT_4_Chana pipeline end to end, in Layer order.

Two stages, in this order:
  1. Self-test every src/ module (its own `if __name__ == "__main__"` block).
     If a module's own guarantees are broken, no notebook result built on
     top of it can be trusted - so this stage runs first and any failure
     here stops the run before touching notebooks.
  2. Rebuild and freshly execute every pipeline notebook, in the same
     dependency order the 7 layers were developed in - not alphabetical,
     since e.g. the downstream experiment (Day 22) depends on files the
     RODARE arm (Day 23) does not, and a later notebook re-reading a stale
     .ipynb from disk would defeat the point of a reproduction gate.

Each notebook is rebuilt from its build_*.py source of truth, then executed
in a fresh kernel via nbclient - the same reproduction-gate approach used
to close Day 24, now a permanent, rerunnable script instead of one-off.

Usage: python run_all.py
"""
import subprocess
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PIPELINE_DIR = ROOT / "notebooks" / "pipeline"

SRC_MODULES = [
    "data/split.py",
    "data/loader.py",
    "data/guard.py",
    "data/furniture.py",
    "data/pyramid.py",
    "data/patches.py",
    "data/claims.py",
    "data/augmentation.py",
    "models/denoiser.py",
    "models/sampler.py",
    "models/patch_classifier.py",
    "evaluation/metrics.py",
    "evaluation/copying.py",
    "evaluation/label_validity.py",
]

# (build script, notebook filename, layer/day label) - dependency order, not alphabetical
PIPELINE_NOTEBOOKS = [
    ("build_source_safe_split_and_loader_notebook.py", "source_safe_split_and_loader.ipynb", "Layer 1"),
    ("build_data_pipeline_validation_notebook.py", "data_pipeline_validation.ipynb", "Layer 1"),
    ("build_multiscale_patch_pipeline_notebook.py", "multiscale_patch_pipeline.ipynb", "Layer 2"),
    ("build_approximate_denoiser_notebook.py", "approximate_denoiser.ipynb", "Layer 3"),
    ("build_denoiser_real_patches_notebook.py", "denoiser_real_patches.ipynb", "Layer 3"),
    ("build_single_scale_sampler_verification_notebook.py", "single_scale_sampler_verification.ipynb", "Layer 4"),
    ("build_coarse_to_fine_sampler_notebook.py", "coarse_to_fine_sampler.ipynb", "Layer 5"),
    ("build_exact_vs_fast_config_notebook.py", "exact_vs_fast_config.ipynb", "Layer 6"),
    ("build_efficiency_profiling_notebook.py", "efficiency_profiling.ipynb", "Layer 6"),
    ("build_basic_evaluator_notebook.py", "basic_evaluator.ipynb", "Layer 6"),
    ("build_copying_diagnostic_notebook.py", "copying_diagnostic.ipynb", "Day 20"),
    ("build_label_validity_notebook.py", "label_validity.ipynb", "Day 20/21"),
    ("build_downstream_experiment_setup_notebook.py", "downstream_experiment_setup.ipynb", "Day 21"),
    ("build_downstream_experiment_run_notebook.py", "downstream_experiment_run.ipynb", "Day 22"),
    ("build_rodare_arm_notebook.py", "rodare_arm.ipynb", "Day 23"),
]


def run_src_self_tests():
    print("=" * 70)
    print("STAGE 1: src/ module self-tests")
    print("=" * 70)
    failures = []
    for rel_path in SRC_MODULES:
        module_path = SRC / rel_path
        t0 = time.perf_counter()
        result = subprocess.run(
            [sys.executable, str(module_path)],
            cwd=module_path.parent,
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - t0
        status = "PASS" if result.returncode == 0 else "FAIL"
        print(f"  [{status}] {rel_path} ({elapsed:.1f}s)")
        if result.returncode != 0:
            failures.append(rel_path)
            print(result.stdout[-2000:])
            print(result.stderr[-2000:])
    return failures


def run_pipeline_notebooks():
    print("=" * 70)
    print("STAGE 2: pipeline notebooks (rebuild + fresh execute)")
    print("=" * 70)
    failures = []
    for build_script, notebook_name, layer in PIPELINE_NOTEBOOKS:
        print(f"  building {notebook_name} ({layer}) ...")
        build_result = subprocess.run(
            [sys.executable, build_script],
            cwd=PIPELINE_DIR,
            capture_output=True,
            text=True,
        )
        if build_result.returncode != 0:
            print(f"  [FAIL] {notebook_name}: build script failed")
            print(build_result.stderr[-2000:])
            failures.append(notebook_name)
            continue

        notebook_path = PIPELINE_DIR / notebook_name
        t0 = time.perf_counter()
        try:
            nb = nbformat.read(notebook_path, as_version=4)
            NotebookClient(nb, timeout=600, kernel_name="python3").execute()
            nbformat.write(nb, notebook_path)
            elapsed = time.perf_counter() - t0
            print(f"  [PASS] {notebook_name} ({layer}, {elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"  [FAIL] {notebook_name} ({layer}, {elapsed:.1f}s): {e}")
            failures.append(notebook_name)
    return failures


def main():
    start = time.perf_counter()
    src_failures = run_src_self_tests()
    if src_failures:
        print()
        print(f"Stopping before notebooks: {len(src_failures)} src module(s) failed self-test: {src_failures}")
        sys.exit(1)

    notebook_failures = run_pipeline_notebooks()
    total = time.perf_counter() - start

    print("=" * 70)
    print(f"DONE in {total:.1f}s")
    print(f"  src/ modules:  {len(SRC_MODULES)}/{len(SRC_MODULES)} passed")
    if notebook_failures:
        print(f"  notebooks:     {len(PIPELINE_NOTEBOOKS) - len(notebook_failures)}/{len(PIPELINE_NOTEBOOKS)} passed")
        print(f"  FAILED: {notebook_failures}")
        sys.exit(1)
    else:
        print(f"  notebooks:     {len(PIPELINE_NOTEBOOKS)}/{len(PIPELINE_NOTEBOOKS)} passed")
        print("  Full reproduction gate: GREEN")


if __name__ == "__main__":
    main()
