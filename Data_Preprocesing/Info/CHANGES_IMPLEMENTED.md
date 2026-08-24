# Safety and Reliability Changes

The preprocessing implementation was corrected on 2026-08-21.

- Removed automatic contour/central-ROI annotations from ingestion.
- Added strict `annotation.mode: require` as the default.
- Limited the RGB baseline to RGB imagery and excluded RGB-D depth maps.
- Excluded named published flip/warp variants by default and added provenance grouping support.
- Rejected empty or invalid annotations instead of treating them as usable samples.
- Rejected unknown VOC/COCO class names rather than silently mapping them to class `0`.
- Corrected bounding-box clipping to operate on corners rather than independently clamping centre and size.
- Seeded Python, NumPy, and OpenCV RNGs for repeatable runs.
- Made failed image reads/writes fatal and added final YOLO output integrity validation.
- Prevented accidental deletion of an existing processed dataset unless explicitly enabled in configuration.
- Populated split-level statistics.

The next prerequisite is human-reviewed bounding-box annotation. No training result should be interpreted until that prerequisite is complete.
