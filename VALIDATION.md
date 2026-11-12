# Validation

Validated on Windows with Python 3.13 using the pinned dependencies.

- Complete supplied input: 750 frames, 1920 × 1080, 25 fps, 30 seconds.
- Fresh detection and pitch inference on all 750 frames; no tracking cache.
- Output decoded successfully: 750 frames at 25 fps.
- Animated README preview: 50 frames, 640 × 360, first six seconds of a separate successful preview run using the same pipeline.
- Three regression tests passed: sparse player clustering, homography projection/fallback, and renderer team/shared-transform handling.
- Python source compilation passed.
- Both supplied custom YOLO checkpoints loaded and performed inference.

This verifies execution and file integrity, not model accuracy. No labelled test dataset was supplied. Visual checks covered the start and middle of the README preview. A clean-environment dependency installation was not performed.
