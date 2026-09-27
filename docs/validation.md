# Validation on this machine

Latest run: 2026-09-28, Ubuntu 22.04 x86_64, SDK wheel 2.53.1.4623.

- 9 unit tests passed; Ruff checks passed.
- Official LeRobot v3 loader read all 74 frames across 2 synthetic episodes; all 74 MP4 frames decoded. These are software test episodes, not physical demonstrations.
- D435 `243622071623`, firmware 05.17.00.10: RGB snapshot captured at `artifacts/d435_overhead.png`; custom UMI assembly visible.
- D405 `352122273221`, firmware 05.15.01.55: RGB and raw Z16 depth streaming works. Finger markers 13 and 14 detected in DICT_4X4_50. Intrinsics report inverse Brown–Conrady distortion; corner rectification handles it through RealSense deprojection.
- T265 `943222111495`, firmware 0.2.0.951: simultaneous pose, gyro and accelerometer capture works after USB access became available.
- Viser startup and camera preview updates completed during a D405-only capture.

Latest simultaneous diagnostic capture: `artifacts/combined_hardware_test_dedup`. Full metrics are in its `test-report.json`.

| Stream | Unique samples | Counter gaps | Approximate timing jitter p95–p05 |
|---|---:|---:|---:|
| D405 RGB | 268 | 19 | 42.3 ms |
| D405 depth | 272 | 28 | 44.4 ms |
| T265 pose | 1905 | 0 | 1.49 ms |
| T265 gyro | 1906 | 0 | 1.58 ms |
| T265 accel | 594 | 0 | 2.07 ms |

The run spans approximately 9.5 seconds of RGB/T265 data after startup and 10 seconds of depth. RealSense synchronized tracking framesets repeat older samples as individual streams advance; the recorder now persists each sensor frame number only once. No duplicates or counter gaps occurred in the final T265 logs.

**Hardware data-quality acceptance has NOT passed.** T265 tracker confidence was 2 for every retained sample (default required confidence is 3). RGB/depth still show frame loss and excessive timing jitter. A comparison with an existing modern SDK also showed camera frame loss; this does not establish a single root cause. Independent video sensor callbacks eliminated repeated video deliveries but did not eliminate loss.

Outstanding: resolve camera timing/frame loss under representative load, obtain confidence-3 tracking, measure marker size and jaw aperture mapping, calibrate tool extrinsics and time offset, validate physical repeatability, then capture and verify a real accepted LeRobot demonstration. No fabricated physical calibration or synthetic pose has been attached to real camera images as training data.
