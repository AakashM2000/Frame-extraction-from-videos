# Verification

Executed in the development environment on 20 October 2026, using Python 3.12, PyTorch 2.14.1, NumPy 2.5.4, SciPy 1.17.0 and PyAV 19.0.1. These are observed test versions, not a guarantee for every supported dependency combination.

## Automated checks

`python -m unittest discover -s tests -v`: **12 tests passed**.

- CVAT label/color mapping and unknown-color rejection.
- Explicit validation-run splitting and empty-split rejection.
- Retention of reviewed background-only frames.
- Known circle-radius recovery, short-arc rejection and straight-line rejection.
- Solid-contact exclusion from meniscus fitting.
- Synthetic five-pixel film thickness; rejection when the upper interface contacts the tool.
- Translation recovery from a textured patch and rejection of a textureless frame.
- Quadratic-motion acceleration with irregular timestamps and no fitting across missing-data gaps.
- Odd-sized U-Net output dimensions, loss backward propagation, all-ignore loss handling, checkpoint reload and tiled native-size inference.

## End-to-end smoke test

`python tests/smoke_workflow.py`: **passed**.

- Built two independent synthetic groups with four reviewed frames total.
- Ran one CPU training epoch with a batch size larger than the training set.
- Reloaded `last.pt` and resumed through epoch two.
- Encoded a 12-frame video at 25 fps and extracted timestamped annotation images.
- Ran tiled segmentation on all 12 frames, saved original-size masks and encoded an overlay MP4.
- Confirmed 12 CSV rows with final time 0.44 seconds.
- Confirmed unsupported-view radii/thickness remain unavailable.
- Recovered a synthetic raster-circle radius to within one pixel.

CPU execution emitted sandbox-specific NNPACK initialization warnings but completed successfully. No GPU execution or interactive desktop selector was tested here. The temporary synthetic checkpoint is not included as a trained coating model.

## Scope

These tests establish software behavior on controlled data. They do **not** establish blade-coating segmentation accuracy, optical resolution, film-thickness accuracy, physical curvature accuracy or real-time performance. Those require the user's videos, reviewed masks, calibration and independent reference measurements. No supplied blade-coating video was available for that validation.
