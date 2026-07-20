# Geodesic Real-Robot Summary Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate the revised two-row real-robot summary figure and six matching standalone panels.

**Architecture:** Reuse the force metric extractor in one plotting entry point. Use a four-column GridSpec for the 13 x 5.1 in combined figure, then redraw the same axes into standalone 6.5 x 2.55 in process figures and 3.25 x 2.55 in boxplots.

**Tech Stack:** Python, NumPy, SciPy, matplotlib, pytest.

---

### Task 1: Add outward correction metric

**Files:**
- Modify: `EXPERIMENT/geodesic_real_robot/force_analysis.py`
- Modify: `tests/test_geodesic_real_robot_force_analysis.py`

- [ ] Add a failing assertion for cumulative positive offset increments.
- [ ] Run the focused test and verify it fails.
- [ ] Add `force_offset_outward_motion_mm` to `compute_force_metrics`.
- [ ] Run the focused test and verify it passes.

### Task 2: Add summary figure

**Files:**
- Create: `EXPERIMENT/geodesic_real_robot/plot_geodesic_summary_figure.py`
- Modify: `tests/test_geodesic_real_robot_force_analysis.py`

- [ ] Add a failing export test for one combined and six standalone PDF/SVG/PNG figures.
- [ ] Implement the two wide process panels and four simulation-style boxplots.
- [ ] Run the export test and verify it passes.
- [ ] Generate the real Group 2-8 figure and inspect output dimensions and files.
