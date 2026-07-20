# Teleoperation Comparison Mode Design

## Purpose

Provide one GUI workflow for paired comparison of the existing Full-joint tracking
and surface-aware Darboux mapping methods. The mode compares complete teleoperation
systems while holding the planned 5 cm surface segment, participant metadata, sensors,
recording format, and retreat behavior constant.

## Planning Workflow

The existing photo capture, segmentation, and path planning workflow is unchanged.
In comparison mode, `Confirm path` only locks the current planned path and does not
move UR5. Geodesic optimization remains available after confirmation.

`Generate trial pair` samples one path-point start with at least 0.05 m of remaining
arc length. Its endpoint is interpolated at exactly 0.05 m along the active path.
The pair remains fixed across participants until the user explicitly regenerates it.

## Participant Workflow

The GUI accepts a participant identifier and resets per-participant completion and
Darboux calibration when confirmed. Both trial-mode buttons are enabled; the GUI
does not enforce which method runs first. The actual sequence index is recorded.

Full-joint preparation moves UR5 to the configured default joint pose, checks that
the current mapped GELLO joint command is close enough for safe absolute-joint
handover, and then reuses the existing joint positioning loop. The operator manually
reaches the displayed segment start and starts recording.

Darboux preparation automatically moves to the segment start with the existing
Cartesian interpolation. A new surface controller is created for each participant.
The participant performs Set neutral, Calibrate +X, and Calibrate +Z before recording.
No prior participant calibration is reused.

## Completion and Recording

The scan length is 0.05 m. Automatic completion requires both:

- probe-tip distance to the interpolated endpoint below 0.005 m;
- nearest path arc length at least endpoint arc length minus 0.003 m.

Manual finish and a 60 s timeout remain available. Every outcome is retained and
tagged as `reached`, `manual`, or `timeout`. Recording stops before retreat so the
retreat does not contaminate scan metrics. Existing normal retreat and default-joint
motion are then reused.

Every frame records participant ID, pair ID, sequence index, teleoperation mode,
segment indices and arc lengths, scan length, and action mode. Start
position/orientation error and approach duration are captured when scanning starts.
The final completion reason and scan duration are written to the episode-level
`comparison_trial_summary.json`.

## Safety

Comparison mode does not modify control gains or controller mathematics. Full-joint
handover is blocked until the maximum joint mismatch is below a configurable threshold.
Only one teleoperation or motion worker may command UR5 at a time.
