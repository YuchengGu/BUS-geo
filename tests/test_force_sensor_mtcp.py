from gello.force_sensor_mtcp import ForceSensorMTCP


class BrokenPipeClient:
    def __init__(self):
        self.closed = False

    def read_holding_registers(self, *args, **kwargs):
        raise BrokenPipeError(32, "Broken pipe")

    def close(self):
        self.closed = True


def test_force_sensor_marks_broken_pipe_invalid_and_disconnects(capsys):
    sensor = ForceSensorMTCP("127.0.0.1", reconnect_interval_s=1000.0, log_interval_s=1000.0)
    client = BrokenPipeClient()
    sensor.client = client
    sensor.is_connected = True

    assert sensor.read_values() is None

    captured = capsys.readouterr()
    assert "Broken pipe" in captured.out
    assert sensor.is_connected is False
    assert sensor.client is None
    assert client.closed is True
    assert sensor.last_metadata["valid"] is False
    assert "Broken pipe" in sensor.last_metadata["error"]

