# Teleoperation Comparison Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a paired Full-joint versus Darboux experiment workflow to the existing GUI without duplicating either controller.

**Architecture:** A new pure-Python comparison state module owns the fixed 5 cm segment, participant, mode, sequence, and endpoint rules. `app.py` remains the orchestration layer and delegates actual control to the existing `TeleopLoop`, movement helpers, recorder, and safe-retreat functions.

**Tech Stack:** Python 3.11, NumPy, Open3D GUI, pytest.

---

### Task 1: Comparison state and fixed-arclength segment

**Files:**
- Create: `visual_guided_collection_gui/comparison_experiment.py`
- Create: `tests/test_comparison_experiment.py`

- [ ] Write failing tests for exact 0.05 m segment generation, pair persistence, participant reset, free mode ordering, and endpoint completion.
- [ ] Run `python -m pytest tests/test_comparison_experiment.py -q` and verify missing-module failure.
- [ ] Implement `ComparisonSegment`, `ComparisonExperiment`, segment interpolation, and completion evaluation.
- [ ] Re-run the focused tests and verify they pass.

### Task 2: CLI and GUI action availability

**Files:**
- Modify: `visual_guided_collection_gui/main.py`
- Modify: `visual_guided_collection_gui/state.py`
- Modify: `tests/test_surface_cartesian_gui_args.py`

- [ ] Add failing tests for `--operation-mode comparison`, comparison defaults, and comparison actions.
- [ ] Run the focused tests and verify the expected failures.
- [ ] Add comparison CLI defaults and action keys without changing demo/auto defaults.
- [ ] Re-run the focused tests and verify they pass.

### Task 3: GUI widgets and planning transition

**Files:**
- Modify: `visual_guided_collection_gui/app.py`
- Modify: `tests/test_surface_random_local.py`

- [ ] Add failing source-level tests that comparison Confirm path does not call motion and geodesic completion preserves the confirmed stage.
- [ ] Add participant input and comparison buttons to the existing manual layout.
- [ ] Add comparison-specific button gating and planning transitions.
- [ ] Re-run focused GUI tests.

### Task 4: Reused Full-joint and Darboux preparation

**Files:**
- Modify: `visual_guided_collection_gui/app.py`
- Modify: `visual_guided_collection_gui/device_manager.py`
- Modify: `tests/test_surface_cartesian_device_manager.py`

- [ ] Add failing tests for joint mismatch calculation and action-mode metadata.
- [ ] Implement safe Full-joint handover using existing positioning.
- [ ] Implement Darboux start motion using existing random-local target and Cartesian movement.
- [ ] Reset the Darboux controller at participant confirmation and retain existing calibration callbacks.
- [ ] Re-run focused tests.

### Task 5: Recording, automatic/manual completion, and retreat

**Files:**
- Modify: `visual_guided_collection_gui/app.py`
- Modify: `visual_guided_collection_gui/collection_session.py`
- Modify: `visual_guided_collection_gui/episode_recorder.py`
- Modify: `tests/test_surface_cartesian_collection_session.py`
- Modify: `tests/test_visual_guided_collection_gui.py`

- [ ] Add failing tests for comparison metadata, timeout/reached/manual outcomes, and one-shot completion signaling.
- [ ] Start the existing joint or surface recorder according to the selected mode.
- [ ] Evaluate endpoint completion from recorded enriched observations and post completion to the GUI thread.
- [ ] Stop recording before invoking the existing safe-retreat flow.
- [ ] Preserve the pair while resetting participant completion for the next participant.
- [ ] Re-run focused tests.

### Task 6: Regression verification

**Files:**
- Verify only.

- [ ] Run comparison, random-local, Cartesian, autoscan, BO, and general GUI tests.
- [ ] Run `python -m py_compile` on modified modules.
- [ ] Run `git diff --check`.
- [ ] Review the final diff for unrelated changes and report hardware-test limitations.
