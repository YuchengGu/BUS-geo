from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from visual_guided_collection_gui.app import VisualGuidedCollectionApp


def test_safe_position_retreat_still_moves_to_safe_joints_after_tcp_retreat_timeout():
    app = VisualGuidedCollectionApp.__new__(VisualGuidedCollectionApp)
    calls: list[str] = []
    app._post_status = lambda _message: None
    app._retreat_tcp_along_path_normal = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        TimeoutError("TCP settle timeout")
    )
    app._move_to_safe_joint_position = lambda **_kwargs: calls.append("safe_joints")
    path = SimpleNamespace(
        positions_base=np.array([[0.0, 0.0, 0.0]]),
        normals_base=np.array([[0.0, 0.0, 1.0]]),
    )

    warning = app._run_safe_position_retreat(path=path, pose_index=0, reason="Safe stop")

    assert calls == ["safe_joints"]
    assert isinstance(warning, TimeoutError)


def test_auto_scan_retreat_marks_safe_position_reached_even_with_tcp_warning():
    app = VisualGuidedCollectionApp.__new__(VisualGuidedCollectionApp)
    warning = TimeoutError("TCP settle timeout")
    app._auto_scan_safe_position_reached = False
    app._run_safe_position_retreat = lambda **_kwargs: warning
    path = SimpleNamespace(positions_base=np.zeros((3, 3)))
    result = SimpleNamespace(last_pose_index=2)

    returned_warning = app._run_auto_scan_safe_retreat(path, result)

    assert returned_warning is warning
    assert app._auto_scan_safe_position_reached is True


def test_safe_stop_waits_for_auto_worker_instead_of_starting_competing_motion():
    class FinishingThread:
        def __init__(self):
            self.alive = True
            self.join_timeout = None

        def is_alive(self):
            return self.alive

        def join(self, timeout):
            self.join_timeout = timeout
            self.alive = False

    app = VisualGuidedCollectionApp.__new__(VisualGuidedCollectionApp)
    thread = FinishingThread()
    app._auto_scan_thread = thread
    app._post_status = lambda _message: None

    app._wait_for_auto_scan_worker_before_safe_stop()

    assert thread.join_timeout == 10.0
    assert thread.is_alive() is False


def test_safe_stop_refuses_concurrent_motion_when_auto_worker_does_not_finish():
    class StuckThread:
        def is_alive(self):
            return True

        def join(self, timeout):
            assert timeout == 10.0

    app = VisualGuidedCollectionApp.__new__(VisualGuidedCollectionApp)
    app._auto_scan_thread = StuckThread()
    app._post_status = lambda _message: None

    try:
        app._wait_for_auto_scan_worker_before_safe_stop()
    except TimeoutError as exc:
        assert "still running" in str(exc)
    else:
        raise AssertionError("Expected a timeout instead of concurrent safe motion")
