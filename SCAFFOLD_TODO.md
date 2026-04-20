# M-105 scaffold TODO

Scaffolded from template: `M-041_multibypass_phase_recognition_data-pipeline` (2026-04-20)

## Status
- [x] config.py updated (domain=pitvis_surgical_phase_recognition, s3_prefix=M-105_PitVis/raw/, fps=3)
- [ ] core/download.py: update URL / Kaggle slug / HF repo_id
- [ ] src/download/downloader.py: adapt to dataset file layout
- [ ] src/pipeline/_phase2/*.py: adapt raw → frames logic (inherited from M-041_multibypass_phase_recognition_data-pipeline, likely needs rework)
- [ ] examples/generate.py: verify end-to-end on 3 samples

## Task prompt
This endoscopic pituitary surgery frame (PitVis). Predict the current surgical phase from the workflow taxonomy.

Fleet runs likely FAIL on first attempt for dataset parsing; iterate based on fleet logs at s3://vbvr-final-data/_logs/.
