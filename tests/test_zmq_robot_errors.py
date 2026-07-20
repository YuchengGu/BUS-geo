import pickle

import numpy as np
import pytest

from gello.zmq_core.robot_node import ZMQClientRobot


class FakeSocket:
    def __init__(self, response):
        self.response = pickle.dumps(response)

    def send(self, _message):
        pass

    def recv(self):
        return self.response


def test_command_joint_state_propagates_server_error():
    client = ZMQClientRobot.__new__(ZMQClientRobot)
    client._socket = FakeSocket({"error": "RuntimeError: RTDE servoJ command rejected"})

    with pytest.raises(RuntimeError, match="servoJ command rejected"):
        client.command_joint_state(np.zeros(6))
