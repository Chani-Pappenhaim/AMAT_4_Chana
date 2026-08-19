# Findings

- EMPS is heterogeneous: 465 images from 322 distinct source papers - not a
  uniform acquisition series. Any statistical baseline built from it should be
  treated as a rough population estimate, not a single-instrument spec.
- No acquisition metadata (dose, scan rate, instrument, accelerating voltage)
  is available per image - only the image and its source DOI. This limits
  confidence on any artifact diagnosis to "medium" at best (pattern-based
  inference, not physically verified cause).
- Per the general team plan: unverified local data must be called "electron
  microscopy," not "SEM," until modality/calibration evidence is produced.
  **EMPS as currently used has not been verified against this bar.**
- Leader review gate answer (recorded, not yet re-derived from our own data):
  a generator that reproduces object shape but not SEM-specific statistics is
  **not necessarily successful** - shape correctness and statistical
  correctness are separate, both-required criteria.
