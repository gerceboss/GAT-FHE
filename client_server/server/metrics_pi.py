"""
Raspberry Pi 4 metrics utilities (Python equivalent of benchutils C++).

Provides:
- RSS measurement
- CPU frequency (Hz)
- CPU voltage (V)
- Power estimation model
- Context-based step recorder
"""

import os
import time
import subprocess
from dataclasses import dataclass
from typing import Optional


# ===============================
# Memory (RSS)
# ===============================

def get_rss_bytes() -> int:
    """Return resident set size in bytes (Linux)."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return int(parts[1]) * 1024  # kB → bytes
    except OSError:
        pass
    return 0


# ===============================
# CPU Frequency (Hz)
# ===============================

def get_cpu_freq() -> int:
    """
    Read ARM frequency in Hz using vcgencmd.
    Equivalent to: vcgencmd measure_clock arm
    """
    try:
        out = subprocess.check_output(
            ["vcgencmd", "measure_clock", "arm"],
            encoding="utf-8"
        ).strip()
        # format: frequency(48)=1500000000
        if "=" in out:
            return int(out.split("=")[1])
    except Exception:
        pass
    return 0


# ===============================
# CPU Voltage (V)
# ===============================

def get_cpu_volt() -> float:
    """
    Read core voltage using vcgencmd.
    Equivalent to: vcgencmd measure_volts core
    """
    try:
        out = subprocess.check_output(
            ["vcgencmd", "measure_volts", "core"],
            encoding="utf-8"
        ).strip()
        # format: volt=0.8625V
        if "=" in out:
            val = out.split("=")[1].replace("V", "")
            return float(val)
    except Exception:
        pass
    return 0.0


# ===============================
# Raspberry Pi 4 Power Model
# ===============================

def estimate_power(
    volt: float,
    freq_hz: int,
    active_cores: float = 1.0,
) -> float:
    """
    Dynamic power model for Raspberry Pi 4.

    P = idle + C * V^2 * f * cores
    """
    C = 3.0e-9   # Calibration constant (empirical)
    idle = 1.2   # Idle baseline watts (Pi4)
    return idle + C * (volt ** 2) * freq_hz * active_cores


# ===============================
# Step Metric Dataclass
# ===============================

@dataclass
class StepMetric:
    name: str
    seconds: float
    rss_delta_bytes: int
    rss_after_bytes: int
    freq_hz: int
    volt: float
    power_watts: float
    energy_joules: float


# ===============================
# Metrics Recorder
# ===============================

class MetricsRecorder:
    def __init__(self) -> None:
        self._metrics: list[StepMetric] = []

    def step(self, name: str, active_cores: float = 1.0):
        return _StepContext(self, name, active_cores)

    def add(self, metric: StepMetric) -> None:
        self._metrics.append(metric)

    def to_dict(self) -> dict:
        return {
            m.name: {
                "seconds": m.seconds,
                "rss_delta_bytes": m.rss_delta_bytes,
                "rss_after_bytes": m.rss_after_bytes,
                "freq_hz": m.freq_hz,
                "volt": m.volt,
                "power_watts": m.power_watts,
                "energy_joules": m.energy_joules,
            }
            for m in self._metrics
        }

    def write_txt(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "step,seconds,rss_delta_bytes,rss_after_bytes,"
                "freq_hz,volt,power_watts,energy_joules\n"
            )
            for m in self._metrics:
                f.write(
                    f"{m.name},{m.seconds:.6f},"
                    f"{m.rss_delta_bytes},{m.rss_after_bytes},"
                    f"{m.freq_hz},{m.volt:.4f},"
                    f"{m.power_watts:.6f},{m.energy_joules:.6f}\n"
                )


# ===============================
# Context Manager for Steps
# ===============================

class _StepContext:
    def __init__(self, rec: MetricsRecorder, name: str, active_cores: float):
        self._rec = rec
        self._name = name
        self._active_cores = active_cores

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._rss0 = get_rss_bytes()
        self._freq0 = get_cpu_freq()
        self._volt0 = get_cpu_volt()
        return self

    def __exit__(self, exc_type, exc, tb):
        t1 = time.perf_counter()
        rss1 = get_rss_bytes()

        dt = t1 - self._t0
        rss_delta = rss1 - self._rss0

        # Use mid-step values (better than start-only)
        freq = get_cpu_freq()
        volt = get_cpu_volt()

        power = estimate_power(
            volt,
            freq,
            active_cores=self._active_cores,
        )

        energy = power * dt  # Joules

        self._rec.add(
            StepMetric(
                name=self._name,
                seconds=dt,
                rss_delta_bytes=rss_delta,
                rss_after_bytes=rss1,
                freq_hz=freq,
                volt=volt,
                power_watts=power,
                energy_joules=energy,
            )
        )