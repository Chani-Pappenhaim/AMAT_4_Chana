"""Plain dataclasses: single source of truth for the shapes that flow
between L1 (data spine) and every later layer. No behavior here - just
field definitions, so a change to what a manifest row or a config carries
happens in exactly one place.
"""
from dataclasses import dataclass
from typing import Optional

# A DOI (EMPS), a field id (RODARE), or an intensity-set id (NIST) - never
# a filename or a tile. See data/grouping.py for the per-dataset mapping.
GroupID = str


@dataclass(frozen=True)
class SourceManifestRow:
    source_id: str
    dataset: str  # "emps" | "nist" | "rodare"
    group_id: GroupID
    split: str  # "train" | "validation" | "test"
    image_path: str
    label_path: Optional[str] = None
    modality_claim: str = "electron microscopy"  # "SEM" only ever set on rodare rows
    height: int = 0
    width: int = 0
    instance_count: int = 0
    furniture_fraction: float = 0.0


@dataclass(frozen=True)
class PatchBankRecord:
    source_id: str
    scale_index: int
    patch_index: int
    row: int
    col: int
    patch_size: int


@dataclass
class GenerationConfig:
    backend: str = "exact"  # "exact" | "approximate"
    num_scales: int = 1
    steps_per_scale: int = 1
    patch_size: int = 7
    seed: int = 0


if __name__ == "__main__":
    row = SourceManifestRow(
        source_id="x", dataset="emps", group_id="doiA", split="train", image_path="x.png",
    )
    assert row.modality_claim == "electron microscopy"
    patch = PatchBankRecord(source_id="x", scale_index=0, patch_index=0, row=0, col=0, patch_size=7)
    config = GenerationConfig()
    assert config.backend == "exact"
    print("schemas: SourceManifestRow, PatchBankRecord, GenerationConfig construct with expected defaults")
