from backend.telemetry import Telemetry


def test_metrics_uses_hottest_cpu_and_excludes_other_devices():
    payload = """# TYPE lhm_cpu_temperature_celsius gauge
lhm_cpu_temperature_celsius {"sensorName"="CPU (Tctl/Tdie)", "hardwareId"="/amdcpu/0"} 61.5
lhm_cpu_temperature_celsius{"sensorName"="CCD1"} 58.25
lhm_gpu_temperature_celsius{"sensorName"="GPU"} 99.0
lhm_motherboard_temperature_celsius{"sensorName"="Board"} 100.0
"""
    assert Telemetry._parse_lhm_metrics(payload) == 61.5


def test_invalid_or_missing_cpu_metrics_remain_unavailable():
    for payload in [
        "",
        "# No sensors",
        "lhm_cpu_temperature_celsius{} NaN",
        "lhm_cpu_temperature_celsius{} 0",
        "lhm_cpu_temperature_celsius{} 999",
        "lhm_gpu_temperature_celsius{} 61",
    ]:
        assert Telemetry._parse_lhm_metrics(payload) is None


def test_unavailable_http_does_not_create_a_temperature(monkeypatch):
    class Offline:
        def open(self, *args, **kwargs):
            raise OSError("No local sensor feed")

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Offline())
    assert Telemetry._lhm_http_temperature() is None
