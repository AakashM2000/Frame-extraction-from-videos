# Review of the submitted scripts

Reviewed the five latest uploaded `(1).py` files and their dependency pattern. This package is a fresh standalone replacement; original uploads are preserved. The spin-coater reference is a different experimental geometry and its learned classes/weights are not assumed transferable.

| Submitted module | Finding | Replacement/change |
|---|---|---|
| `__init__.py` | Reliance on external segmentation components/class conventions | One documented four-class definition plus ignore ID |
| `video_preprocessing.py` | Independent dimension rounding can change geometry; ROI/output checks and timestamp assumptions need tightening | Native PNGs, explicit timestamps and optional known FPS override, output checks, extraction manifest |
| `dataset.py` | Dropping all-background masks removes useful negatives; unknown colors can conceal errors; filename grouping can leak adjacent frames into validation | Explicit reviewed list, strict name/color mapping, retained negatives, source hashes and explicit acquisition-run groups |
| `train.py` | External model/loss dependency; dropping incomplete batches can produce zero updates; compatible tensor shapes do not imply compatible semantic classes; incomplete experiment bookkeeping | Standalone U-Net and loss, all batches retained, no automatic foreign classifier transfer, full checkpoints and split/settings validation |
| `inference.py` | Mask centroids and class extents do not establish physical speed, film thickness or curvature | Separate textured-feature tracking, timestamp-aware derivatives, selected free-surface circle fits, visible-layer thickness and rejection statuses |

## Design choices

- U-Net is implemented directly in PyTorch; no legacy TensorFlow/jakeret package or spin-coater source installation is required.
- Native-scale crops train the network; overlapping native-scale tiles run inference. Full-frame validation uses more memory but avoids geometry changes.
- Explicit run-group splitting replaces inferred grouping. No automatic single-video random split is offered as independent validation.
- The extractor uses predictable time sampling. Review diversity manually; it does not reproduce the submitted extractor's duplicate/diversity filtering.
- Coating includes the visible meniscus and flat wet film. Separate geometric ROIs distinguish the quantities to measure.
- Motion is tracked from rigid features, not a changing liquid-mask centroid. A segmentation model alone does not identify physical coating speed.
- Physical calibration is optional; unavailable/invalid estimates remain blank with status fields. Default calibration and ROIs are intentionally unset.
- This is an offline research pipeline. No actuator commands, PID controller, hardware synchronization, online inference guarantee or industrial certification is provided.

## Remaining experimental work

Train on your CVAT masks; assess on independent runs; check segmentation around thin interfaces; quantify calibration and optical uncertainty; validate motion against known travel/encoder data. The UI selector is designed for a desktop and has not been exercised with your camera imagery. No claim is made that a circle fits your meniscus or that your film is optically resolved.

Reference project: https://gitlab.kit.edu/sven.fischer/spincoater_monitoring/-/tree/main
