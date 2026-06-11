import argparse
import importlib.util
import pickle
import sys
import types
from pathlib import Path

import numpy as np


def load_hand_eye_module(monkeypatch):
    fake_cv2 = types.SimpleNamespace(
        COLOR_RGB2GRAY=0,
        CALIB_HAND_EYE_TSAI=0,
        SOLVEPNP_ITERATIVE=0,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    spec = importlib.util.spec_from_file_location(
        "hand_eye_from_pkl_for_test",
        Path("hand_eye_calibration/hand_eye_from_pkl.py"),
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_is_usable_frame_uses_selected_camera_name(monkeypatch):
    module = load_hand_eye_module(monkeypatch)
    data = {
        "Orbbec_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "ee_pos_rotvec": np.zeros(6),
        "meta": {"modalities": {"Orbbec": {"valid": True, "frame_new": True}}},
    }

    assert module.is_usable_frame(data, require_new_frame=True, camera_name="Orbbec")
    assert not module.is_usable_frame(data, require_new_frame=True, camera_name="D405")


def test_collect_detections_reads_orbbec_rgb_and_metadata(tmp_path, monkeypatch):
    module = load_hand_eye_module(monkeypatch)
    rgb = np.full((4, 5, 3), 17, dtype=np.uint8)
    frame = {
        "Orbbec_rgb": rgb,
        "ee_pos_rotvec": np.arange(6, dtype=float),
        "meta": {
            "sample_index": 12,
            "modalities": {"Orbbec": {"valid": True, "frame_new": True, "frame_id": 34}},
        },
    }
    with open(tmp_path / "000.pkl", "wb") as f:
        pickle.dump(frame, f)

    corners = np.zeros((4, 1, 2), dtype=np.float32)
    monkeypatch.setattr(module, "find_chessboard", lambda image, pattern_size: (True, corners))
    args = argparse.Namespace(
        episode_dir=str(tmp_path),
        board_cols=2,
        board_rows=2,
        stride=1,
        max_frames=40,
        require_new_frame=True,
        camera_name="Orbbec",
    )

    detections, pattern_size = module.collect_detections(args)

    assert pattern_size == (2, 2)
    assert len(detections) == 1
    np.testing.assert_array_equal(detections[0]["rgb"], rgb)
    np.testing.assert_allclose(detections[0]["ee_pos_rotvec"], np.arange(6, dtype=float))
    assert detections[0]["frame_id"] == 34
    assert detections[0]["camera_name"] == "Orbbec"
