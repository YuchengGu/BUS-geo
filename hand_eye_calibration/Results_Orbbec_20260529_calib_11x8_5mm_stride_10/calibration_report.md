# Orbbec Hand-Eye Calibration Report

- episode: `/home/ubuntu22/bc_data/gello/0529_133158`
- camera: `Orbbec`
- selected frames: 97
- board inner corners: 11 x 8
- square size: 0.005 m
- camera calibration RMS: 0.0988 px

## T_tcp_camera

```text
[[ 0.08929534  0.99573184 -0.02333321  0.17384053]
 [-0.995988    0.08940694  0.00378192  0.01613165]
 [ 0.00585193  0.02290189  0.99972059  0.05253159]
 [ 0.          0.          0.          1.        ]]
```

## Validation

```json
{
  "num_pnp_frames": 97,
  "base_board_translation_mean_m": [
    0.01671379813928113,
    -0.406825949592679,
    -0.03652140195570836
  ],
  "base_board_translation_error_mm": {
    "mean": 6.524265866514248,
    "p50": 5.959346315332552,
    "p95": 12.136129297841372,
    "max": 15.462035760646728
  },
  "base_board_rotation_error_deg_vs_first": {
    "mean": 2.2341954849777985,
    "p50": 1.696252706381628,
    "p95": 5.249023426365994,
    "max": 8.708252472744517
  },
  "reprojection_error_px": {
    "mean": 0.09119584603407949,
    "p50": 0.07904251664876938,
    "p95": 0.1870864659547805,
    "max": 0.21661272644996643
  }
}
```
