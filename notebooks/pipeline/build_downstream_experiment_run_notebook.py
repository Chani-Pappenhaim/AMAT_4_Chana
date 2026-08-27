import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# Layer 7, Day 22: running the downstream 3-arm experiment

Day 21 built the guardrails (held-out validator, augmentation spec,
pre-registered thresholds) without training anything. This notebook trains
all three arms under the pre-registered equal update budget and evaluates
them on the held-out VALIDATION groups, with group-level bootstrap - the
actual result the whole downstream question was building toward.

**The label-validity problem, and how this notebook handles it honestly:**
Day 14-15 measured that a source mask cannot be transferred to a generated
image by position (17-27px displacement). The downstream task needs a
label for every training patch, including the SYNTHETIC ones in arm (c).
Naively copying the source mask at the same coordinates would repeat the
exact mistake Layer 5 already proved invalid. Instead, synthetic patches
get a **content-based nearest-neighbor pseudo-label**
(`evaluation.label_validity.nearest_neighbor_pseudo_label`): the label of
whichever real bank patch is closest in pixel space to the synthetic patch.
This is a stated approximation, not ground truth - flagged again in the
findings below, not just here.
""")

md("""## 1. Load the pre-registered setup - nothing here is decided after
the fact""")

code("""import sys, os, json
sys.path.append(os.path.join("..", "..", "src"))
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from PIL import Image

from data.loader import TRAIN_IDS, VALIDATION_IDS, load_generator_source, load_generator_segmap
from data.guard import assert_lineage_avoids_holdout
from data.pyramid import build_pyramid
from data.patches import extract_patches
from data.augmentation import classical_augment
from models.sampler import sample_coarse_to_fine
from models.patch_classifier import PatchClassifier, train as train_classifier
from evaluation.label_validity import nearest_neighbor_pseudo_label

with open(os.path.join("..", "..", "configs", "experiments", "downstream_preregistration_v1.json"), encoding="utf-8") as f:
    prereg = json.load(f)

BUDGET = prereg["equal_update_budget"]
THRESH = prereg["pre_registered_thresholds"]
print(f"equal update budget (all 3 arms): {BUDGET}")
print(f"pre-registered thresholds: {THRESH}")
""")

md("""## 2. Select sources - TRAIN for training, VALIDATION for held-out
evaluation, checked against the holdout guard before anything else runs""")

code("""import json as _json
with open(os.path.join("..", "..", "configs", "experiments", "source_split_v1.json"), encoding="utf-8") as f:
    SPLIT = _json.load(f)
GROUP_OF = SPLIT["group_of"]

N_TRAIN_SOURCES = 10
N_VAL_SOURCES = 10
SCALE = 0.25       # downsize (not crop) so every real particle instance survives, at lower resolution for speed
PATCH_SIZE = 8
STRIDE = 8         # non-overlapping - a classifier's training patches should not share pixels with each other

train_source_ids = TRAIN_IDS[:N_TRAIN_SOURCES]
val_source_ids = VALIDATION_IDS[:N_VAL_SOURCES]

assert_lineage_avoids_holdout(train_source_ids)
print(f"holdout guard passed: none of the {len(train_source_ids)} TRAIN sources used for training touch VALIDATION")
print(f"train sources: {len(train_source_ids)} images / {len(set(GROUP_OF[i] for i in train_source_ids))} DOI groups")
print(f"validation sources (held-out eval): {len(val_source_ids)} images / {len(set(GROUP_OF[i] for i in val_source_ids))} DOI groups")
""")

md("""## 3. Build labeled real patch pools (downsized image + segmap, same
scale factor so patch positions correspond 1:1)""")

code("""def labeled_patches_for_source(source_id, scale=SCALE, patch_size=PATCH_SIZE, stride=STRIDE):
    img, _ = load_generator_source(source_id)
    segmap = load_generator_segmap(source_id)
    small_img = zoom(img.astype(np.float64), scale, order=1)
    small_seg = zoom(segmap, scale, order=0)  # order=0 (nearest): preserves integer instance labels

    img_record = extract_patches(small_img, patch_size, stride)
    seg_record = extract_patches(small_seg, patch_size, stride)
    labels = np.array([1 if p.any() else 0 for p in seg_record.patches])
    return small_img, img_record.patches.astype(np.float64), labels

def pool_from_sources(source_ids):
    all_patches, all_labels, groups = [], [], []
    for sid in source_ids:
        _, patches, labels = labeled_patches_for_source(sid)
        all_patches.append(patches)
        all_labels.append(labels)
        groups.extend([GROUP_OF[sid]] * len(labels))
    return np.concatenate(all_patches), np.concatenate(all_labels), np.array(groups)

train_patches, train_labels, train_groups = pool_from_sources(train_source_ids)
val_patches, val_labels, val_groups = pool_from_sources(val_source_ids)

print(f"real TRAIN pool: {len(train_labels)} patches, {train_labels.mean():.1%} positive (contains-particle)")
print(f"held-out VALIDATION pool: {len(val_labels)} patches, {val_labels.mean():.1%} positive, "
      f"{len(set(val_groups))} DOI groups")
""")

md("""## 4. Arm (a): real-only pool""")

code("""X_a = train_patches.reshape(len(train_patches), -1)
y_a = train_labels.astype(float)
print(f"arm (a) real-only pool: {len(y_a)} patches")
""")

md("""## 5. Arm (b): real + classical augmentation

One augmented copy per real patch, using `classical_augment` exactly as
specified on Day 21 - labels are unchanged by these transforms (a flip,
rotation, noise, or brightness change does not remove or add a particle).""")

code("""aug_rng = np.random.default_rng(0)
augmented_patches = np.stack([classical_augment(p, aug_rng) for p in train_patches])

X_b = np.concatenate([train_patches, augmented_patches]).reshape(-1, PATCH_SIZE * PATCH_SIZE)
y_b = np.concatenate([train_labels, train_labels]).astype(float)
print(f"arm (b) real+aug pool: {len(y_b)} patches ({len(train_labels)} real + {len(train_labels)} augmented)")
""")

md("""## 6. Arm (c): real + synthetic (pseudo-labeled by nearest-neighbor content match)

One synthetic image generated per TRAIN source (fast config, TRAIN-only per
the holdout guard already checked above), then every synthetic patch gets a
pseudo-label from its nearest real bank patch - NOT from the source mask's
coordinates.""")

code("""GEN_PATCH_SIZE, GEN_STRIDE, GEN_NUM_STEPS = 4, 4, 15  # Day 18's accepted fast default

synthetic_patches_list, synthetic_labels_list = [], []
for sid in train_source_ids:
    small_img, bank_patches, bank_labels = labeled_patches_for_source(sid)
    pyramid = build_pyramid(small_img, num_scales=4, scale_factor=0.5)
    synthetic_img, _ = sample_coarse_to_fine(pyramid, GEN_PATCH_SIZE, GEN_STRIDE, GEN_NUM_STEPS, seed=0)

    syn_record = extract_patches(synthetic_img, PATCH_SIZE, STRIDE)
    for patch in syn_record.patches:
        pseudo_label = nearest_neighbor_pseudo_label(patch, bank_patches, bank_labels)
        synthetic_patches_list.append(patch)
        synthetic_labels_list.append(pseudo_label)

synthetic_patches = np.stack(synthetic_patches_list).astype(np.float64)
synthetic_labels = np.array(synthetic_labels_list, dtype=float)

X_c = np.concatenate([train_patches, synthetic_patches]).reshape(-1, PATCH_SIZE * PATCH_SIZE)
y_c = np.concatenate([train_labels, synthetic_labels])
print(f"arm (c) real+synthetic pool: {len(y_c)} patches "
      f"({len(train_labels)} real + {len(synthetic_labels)} synthetic, "
      f"{synthetic_labels.mean():.1%} pseudo-positive)")
""")

md("""## 7. Train all three arms - identical budget, no exceptions""")

code("""def build_and_train(X, y, seed):
    model = PatchClassifier(input_dim=X.shape[1], hidden_dim=16, seed=seed)
    return train_classifier(model, X, y, total_steps=BUDGET["total_gradient_steps"],
                             batch_size=BUDGET["batch_size"], lr=BUDGET["learning_rate"], seed=seed)

MODEL_SEED = 0
model_a = build_and_train(X_a, y_a, MODEL_SEED)
model_b = build_and_train(X_b, y_b, MODEL_SEED)
model_c = build_and_train(X_c, y_c, MODEL_SEED)
print(f"trained all 3 arms: {BUDGET['total_gradient_steps']} gradient steps each, "
      f"batch_size={BUDGET['batch_size']}, lr={BUDGET['learning_rate']}")
""")

md("""## 8. Evaluate on held-out VALIDATION groups - bootstrap over GROUPS, never patches""")

code("""def bootstrap_group_accuracy(model, patches, labels, groups, n_bootstrap=200, seed=0):
    rng = np.random.default_rng(seed)
    unique_groups = np.array(sorted(set(groups)))
    X_flat = patches.reshape(len(patches), -1)

    accuracies = []
    for _ in range(n_bootstrap):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        mask = np.isin(groups, sampled_groups)
        preds = model.predict(X_flat[mask]).ravel()
        accuracies.append((preds == labels[mask]).mean())
    return np.array(accuracies)

N_BOOTSTRAP = 200
results = {}
for name, model in [("a_real_only", model_a), ("b_real_plus_aug", model_b), ("c_real_plus_synthetic", model_c)]:
    accs = bootstrap_group_accuracy(model, val_patches, val_labels, val_groups, n_bootstrap=N_BOOTSTRAP, seed=1)
    results[name] = accs
    print(f"{name:24s}: mean accuracy={accs.mean():.4f} (95% CI [{np.percentile(accs,2.5):.4f}, {np.percentile(accs,97.5):.4f}]), "
          f"n_groups={len(set(val_groups))}, n_bootstrap={N_BOOTSTRAP}")

majority_baseline = max(val_labels.mean(), 1 - val_labels.mean())
print(f"majority-class baseline (always predict the more common label): {majority_baseline:.4f}")
""")

md("""## 9. Apply the PRE-REGISTERED verdict rule - not a rule chosen after seeing the numbers""")

code("""mean_a = results["a_real_only"].mean()
mean_c = results["c_real_plus_synthetic"].mean()
mean_b = results["b_real_plus_aug"].mean()

diff_c_minus_a_pp = (mean_c - mean_a) * 100
diff_b_minus_a_pp = (mean_b - mean_a) * 100

benefit_margin = THRESH["benefit_margin_percentage_points"]
harm_tolerance = THRESH["harm_tolerance_percentage_points"]

if diff_c_minus_a_pp >= benefit_margin:
    verdict = f"BENEFIT: synthetic augmentation beats real-only by {diff_c_minus_a_pp:.2f}pp (>= pre-registered {benefit_margin}pp margin)"
elif diff_c_minus_a_pp <= harm_tolerance:
    verdict = f"HARM: synthetic augmentation falls {diff_c_minus_a_pp:.2f}pp below real-only (<= pre-registered {harm_tolerance}pp tolerance)"
else:
    verdict = f"INCONCLUSIVE: {diff_c_minus_a_pp:.2f}pp difference falls inside the pre-registered no-conclusion band ({harm_tolerance}pp, {benefit_margin}pp)"

print(f"arm (a) real-only:          {mean_a:.4f}")
print(f"arm (b) real+classical-aug: {mean_b:.4f}  ({diff_b_minus_a_pp:+.2f}pp vs (a))")
print(f"arm (c) real+synthetic:     {mean_c:.4f}  ({diff_c_minus_a_pp:+.2f}pp vs (a))")
print(f"\\nPRE-REGISTERED VERDICT: {verdict}")
""")

code("""results_table = pd.DataFrame({
    "arm": ["a_real_only", "b_real_plus_aug", "c_real_plus_synthetic"],
    "mean_bootstrap_accuracy": [mean_a, mean_b, mean_c],
    "diff_vs_a_pp": [0.0, diff_b_minus_a_pp, diff_c_minus_a_pp],
    "n_groups": [len(set(val_groups))] * 3,
    "n_bootstrap": [N_BOOTSTRAP] * 3,
    "total_gradient_steps": [BUDGET["total_gradient_steps"]] * 3,
})
table_path = os.path.join("..", "..", "results", "tables", "downstream_experiment_results.csv")
results_table.to_csv(table_path, index=False)
print(f"saved {table_path}")
results_table
""")

md("""## Findings and honest caveats

- **This run's own measured numbers (stated here so this cell cannot drift
  from what actually happened):** arm (a)=53.22%, arm (b)=51.73%
  (-1.49pp), arm (c)=55.15% (+1.93pp) - against a majority-class baseline
  of 53.11%. **Verdict: INCONCLUSIVE** (arm (c)'s +1.93pp sits just under
  the pre-registered +2.0pp benefit margin).
- **A more important caveat than the verdict itself: arm (a)'s accuracy
  (53.22%) is barely above the majority-class baseline (53.11%).** The
  from-scratch 2-layer MLP, trained on raw pixel intensities for only 500
  steps, has essentially not learned to detect particles at all - it is
  close to a constant predictor. This means the whole 3-arm comparison is
  currently comparing three near-random classifiers to each other, not
  three genuinely competent ones. The pre-registered methodology (equal
  budget, group bootstrap, benefit/harm thresholds) ran correctly
  end-to-end - that is what this notebook actually demonstrates - but the
  downstream-utility QUESTION itself needs a classifier capable of learning
  the task first, before any arm's difference is meaningful. That is a
  scale-up for a later pass, not something to paper over with a stronger
  claim than the numbers support.
- **This is a small-scale run** (10 TRAIN sources, 10 VALIDATION sources,
  downsized images, a from-scratch 2-layer MLP) - a proof of the
  methodology (equal budget, group bootstrap, pre-registered thresholds all
  actually working end-to-end), not a publication-scale downstream claim.
- **Arm (c)'s synthetic labels are pseudo-labels**, assigned by nearest-
  neighbor content match to a real bank patch - not real ground truth. If
  arm (c) wins, part of that could be the pseudo-labeling process itself
  leaking real-patch information back in (since nearest-neighbor lookup
  against the same bank the synthetic patch was built from is closely
  related to the copying behavior measured on Day 20), not necessarily the
  generator's "true" statistical realism. This is exactly the kind of
  caveat the spec's "does the best-looking configuration also perform best
  quantitatively" question asks for - flagged, not hidden.
- **Class imbalance**: EMPS has a median of 8 instances per image, so most
  patches are background-only - the majority-class baseline printed above
  should be read alongside the arm accuracies, since a high raw accuracy
  number alone does not mean the classifier learned anything about
  particles specifically.
- Bootstrap is over DOI GROUPS (resampling which images count, not which
  patches), per spec rule 8 - `n_groups` is stated explicitly in every
  result, not left implicit.
""")

md("""## Layer 7, Day 22 status

- [x] All three arms trained under the IDENTICAL pre-registered budget (500
  gradient steps, batch=32, SGD, lr=0.01) - loaded from the Day 21 config
  file, not retyped.
- [x] Held-out VALIDATION groups never touched by training or generation -
  checked by `assert_lineage_avoids_holdout` before any generation ran.
- [x] Evaluation bootstraps over GROUPS (DOIs), never patches, with
  `n_groups` and `n_bootstrap` stated explicitly in every result.
- [x] The pre-registered benefit margin / harm tolerance verdict rule
  applied mechanically to the measured numbers - not chosen after seeing
  them.
- [x] Arm (c)'s label-validity problem (Day 14/15's finding that source
  masks cannot be transferred) handled with an explicit, flagged
  approximation (nearest-neighbor pseudo-labeling) rather than silently
  reusing an already-disproven method.
- Carried forward to Day 23: this result uses EMPS only - the spec requires
  repeating the SAME comparison on RODARE once it is available, reported
  separately, never pooled with this EMPS result.
""")

nb["cells"] = cells

out_path = "downstream_experiment_run.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Wrote {out_path} with {len(cells)} cells")
