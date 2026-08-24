"""Source/specimen/DOI-safe train/validation/test split for EMPS.

EMPS ships its own train.csv/test.csv (by image filename). We verified those
are already DOI-safe (zero DOI overlap between train and test) - see
docs/research_log.md, 2026-08-19 entry. We keep that test set untouched and
carve a validation set out of the official train pool, split by DOI (source),
never by individual image, so no source contributes images to two splits.
"""
import json
import os
import numpy as np
import pandas as pd


def build_source_split(emps_dir, out_path, val_fraction=0.15, seed=0):
    metadata = pd.read_csv(os.path.join(emps_dir, "metadata.csv"))
    metadata["image_id"] = metadata["filename"].str.replace(".png", "", regex=False)
    doi_of = dict(zip(metadata["image_id"], metadata["doi"]))

    official_train_ids = pd.read_csv(os.path.join(emps_dir, "train.csv"), header=None)[0].tolist()
    official_test_ids = pd.read_csv(os.path.join(emps_dir, "test.csv"), header=None)[0].tolist()

    train_dois_all = sorted({doi_of[i] for i in official_train_ids if i in doi_of})
    test_dois = sorted({doi_of[i] for i in official_test_ids if i in doi_of})

    overlap = set(train_dois_all) & set(test_dois)
    assert not overlap, f"DOI leakage between official train/test: {overlap}"

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(train_dois_all)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_dois = sorted(shuffled[:n_val].tolist())
    train_dois = sorted(shuffled[n_val:].tolist())

    assert not (set(train_dois) & set(val_dois))
    assert not (set(train_dois) & set(test_dois))
    assert not (set(val_dois) & set(test_dois))

    train_ids = sorted(i for i in official_train_ids if doi_of.get(i) in set(train_dois))
    val_ids = sorted(i for i in official_train_ids if doi_of.get(i) in set(val_dois))
    test_ids = sorted(official_test_ids)

    all_ids = train_ids + val_ids + test_ids
    group_of = {image_id: doi_of[image_id] for image_id in all_ids if image_id in doi_of}

    manifest = {
        "seed": seed,
        "val_fraction": val_fraction,
        "train_sources": train_dois,
        "validation_sources": val_dois,
        "test_sources": test_dois,
        "train_ids": train_ids,
        "validation_ids": val_ids,
        "test_ids": test_ids,
        "group_of": group_of,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    manifest = build_source_split(
        emps_dir=os.path.join("..", "..", "AMAT", "amat4-week1", "emps"),
        out_path=os.path.join("configs", "experiments", "source_split_v1.json"),
    )
    print(f"train: {len(manifest['train_ids'])} images / {len(manifest['train_sources'])} sources")
    print(f"validation: {len(manifest['validation_ids'])} images / {len(manifest['validation_sources'])} sources")
    print(f"test: {len(manifest['test_ids'])} images / {len(manifest['test_sources'])} sources")
