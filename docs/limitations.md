# Limitations

- Diagnostics in D2 were run on a single sample image, not swept across the
  full EMPS dataset - the "no strong artifact" finding is a spot check, not a
  dataset-wide claim.
- No source-safe (source/specimen/DOI) train/test split has been created yet.
  Nothing has been trained or generated against a protected test set, but the
  split itself - required before any generator work - is not yet built.
- No modality/calibration evidence exists for EMPS images; per team-plan
  terminology this data should currently be referred to as "electron
  microscopy," not "SEM."
- No generator (training-free or otherwise) has been built yet - only the
  measurement/diagnostic infrastructure (D1 statistics, D2 artifact analyses).
