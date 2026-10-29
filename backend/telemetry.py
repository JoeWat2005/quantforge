"""Best-effort sensors. Missing temperatures remain null, never guessed."""

import math
import os
import platform
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request

import psutil

GIB = 1024**3


class Telemetry:
    def __init__(self):
        self.cpu_temp = None
        self.last_cpu_probe = -100.0
        self.gpu = None
        self.last_gpu_probe = -100.0
        self.error = None
        psutil.cpu_percent(percpu=True)

    def read(self):
        now = time.monotonic()
        if now - self.last_gpu_probe > 2:
            self.last_gpu_probe = now
            if shutil.which("nvidia-smi"):
                try:
                    raw = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                            "--format=csv,noheader,nounits",
                            "--id=0",
                        ],
                        text=True,
                        timeout=2,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    )
                    parts = raw.strip().split(",")

                    def number(i):
                        try:
                            return float(parts[i].strip())
                        except ValueError:
                            return None

                    self.gpu = {
                        "name": parts[0].strip(),
                        "utilization": number(1),
                        "used_mb": number(2),
                        "total_mb": number(3),
                        "temperature": number(4),
                        "watts": number(5),
                    }
                except (subprocess.SubprocessError, OSError):
                    self.gpu = None
        if now - self.last_cpu_probe > 5:
            self.last_cpu_probe = now
            self.cpu_temp = self._cpu_temperature()
        mem = psutil.virtual_memory()
        cores = psutil.cpu_percent(percpu=True)
        return {
            "cpu_percent": sum(cores) / max(1, len(cores)),
            "cores": cores,
            "logical_cores": psutil.cpu_count(),
            "physical_cores": psutil.cpu_count(logical=False),
            "cpu_temperature": self.cpu_temp,
            "ram_used_gb": mem.used / GIB,
            "ram_available_gb": mem.available / GIB,
            "ram_total_gb": mem.total / GIB,
            "gpu": self.gpu,
            "platform": platform.platform(),
            "timestamp": time.time(),
            "temperature_note": "CPU sensor unavailable. Hardware thermal protection remains essential."
            if self.cpu_temp is None
            else "CPU sensor connected",
        }

    @staticmethod
    def _cpu_temperature():
        if hasattr(psutil, "sensors_temperatures"):
            readings = psutil.sensors_temperatures()
            preferred = [
                v.current for k, values in readings.items() if k in ("k10temp", "coretemp", "zenpower") for v in values
            ]
            if preferred:
                return max(preferred)
        if os.name == "nt":
            # Current LibreHardwareMonitor removed WMI. Its metrics endpoint
            # returns Celsius regardless of the units selected in its window.
            temperature = Telemetry._lhm_http_temperature()
            if temperature is not None:
                return temperature
            # Keep WMI support for older releases. ACPI thermal zones are
            # deliberately excluded: they are not reliable CPU die sensors.
            command = "Get-CimInstance -Namespace root/LibreHardwareMonitor -ClassName Sensor -ErrorAction Stop | Where-Object { $_.SensorType -eq 'Temperature' -and $_.Identifier -match '/(amdcpu|intelcpu)/' } | ForEach-Object { $_.Value }"
            try:
                raw = subprocess.check_output(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                    text=True,
                    timeout=3,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                values = [float(line) for line in raw.splitlines() if line.strip()]
                return max(values) if values else None
            except (subprocess.SubprocessError, OSError, ValueError):
                pass
        return None

    @staticmethod
    def _lhm_http_temperature():
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        try:
            with opener.open("http://127.0.0.1:8085/metrics", timeout=1) as response:
                payload = response.read(2_000_001)
            if len(payload) > 2_000_000:
                return None
            return Telemetry._parse_lhm_metrics(payload.decode("utf-8"))
        except (OSError, urllib.error.URLError, UnicodeError, ValueError):
            return None

    @staticmethod
    def _parse_lhm_metrics(payload):
        readings = []
        pattern = r"^lhm_cpu_temperature_celsius\s*\{[^\n]*\}\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
        for match in re.finditer(pattern, payload, re.MULTILINE):
            value = float(match.group(1))
            if math.isfinite(value) and 0 < value < 150:
                readings.append(value)
        return max(readings) if readings else None
