"""Client-side metrics: time, RSS delta, RSS after, and optional energy/power."""

import time
from dataclasses import dataclass
from typing import Optional


def _rss_bytes() -> int:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except OSError:
        pass
    try:
        import resource

        return int(getattr(resource.getrusage(resource.RUSAGE_SELF), "ru_maxrss", 0)) * 1024
    except Exception:
        return 0


def _energy_uj() -> Optional[int]:
    """
    Best-effort energy reading from Linux powercap interface.
    Returns microjoules if available, else None.
    """
    try:
        import os

        base = "/sys/class/powercap"
        if not os.path.isdir(base):
            return None
        for entry in os.listdir(base):
            path = os.path.join(base, entry, "energy_uj")
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    return int(f.read().strip())
    except Exception:
        return None
    return None


@dataclass
class StepMetric:
    name: str
    seconds: float
    rss_delta_bytes: int
    rss_after_bytes: int
    encrypted: bool = True
    energy_joules: float = 0.0
    power_watts: float = 0.0


class MetricsRecorder:
    def __init__(self) -> None:
        self._metrics: list[StepMetric] = []

    def step(self, name: str, encrypted: bool = True):
        return _StepContext(self, name, encrypted)

    def add(self, metric: StepMetric) -> None:
        self._metrics.append(metric)

    def to_dict(self) -> dict:
        return {
            m.name: {
                "seconds": m.seconds,
                "rss_delta_bytes": m.rss_delta_bytes,
                "rss_after_bytes": m.rss_after_bytes,
                "encrypted": m.encrypted,
                "energy_joules": m.energy_joules,
                "power_watts": m.power_watts,
            }
            for m in self._metrics
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MetricsRecorder":
        rec = cls()
        for name, m in data.items():
            rec.add(
                StepMetric(
                    name=name,
                    seconds=float(m.get("seconds", 0.0)),
                    rss_delta_bytes=int(m.get("rss_delta_bytes", 0)),
                    rss_after_bytes=int(m.get("rss_after_bytes", 0)),
                    encrypted=bool(m.get("encrypted", True)),
                    energy_joules=float(m.get("energy_joules", 0.0)),
                    power_watts=float(m.get("power_watts", 0.0)),
                )
            )
        return rec

    def write_csv(self, path: str) -> None:
        """Write client-side metrics to CSV: step, server_time=0, client_time (per phase: keygen, encrypt, decrypt), rss_*, power, energy, throughput."""
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "server_time", "client_time", "rss_after_bytes", "rss_delta_bytes", "power_watts", "energy_joules", "throughput"])
            for m in self._metrics:
                throughput = (1.0 / m.seconds) if m.seconds > 0 else 0.0
                writer.writerow([
                    m.name, "0.0", f"{m.seconds:.6f}", m.rss_delta_bytes, m.rss_after_bytes,
                    f"{m.power_watts:.6f}", f"{m.energy_joules:.6f}", f"{throughput:.6f}",
                ])

    def print_report(self) -> None:
        if not self._metrics:
            return
        print("\n" + "=" * 80)
        print("=== Client Metrics (time + RSS delta, RSS after) ===")
        print("=" * 80)
        for m in self._metrics:
            mode = "ENC" if m.encrypted else "DEC"
            mb = m.rss_after_bytes / (1024 * 1024) if m.rss_after_bytes else 0.0
            dmb = m.rss_delta_bytes / (1024 * 1024)
            print(
                f"  {m.name:<28} {mode:<6} {m.seconds:>8.4f}s  "
                f"RSS Δ {dmb:>+8.2f} MB  RSS {mb:>8.2f} MB"
            )
        total_time = sum(m.seconds for m in self._metrics)
        total_rss_delta = sum(m.rss_delta_bytes for m in self._metrics) / (1024 * 1024)
        enc_time = sum(m.seconds for m in self._metrics if m.encrypted)
        dec_time = sum(m.seconds for m in self._metrics if not m.encrypted)
        print("-" * 80)
        print(
            f"  {'TOTAL':<28} {'':6} {total_time:>8.4f}s  "
            f"RSS Δ {total_rss_delta:>+8.2f} MB"
        )
        if total_time > 0:
            print()
            print(
                f"  Encrypted ops: {enc_time:>8.4f}s "
                f"({enc_time / total_time * 100:>5.1f}%)"
            )
            print(
                f"  Plaintext ops: {dec_time:>8.4f}s "
                f"({dec_time / total_time * 100:>5.1f}%)"
            )
        print("=" * 80)


class _StepContext:
    def __init__(self, rec: MetricsRecorder, name: str, encrypted: bool) -> None:
        self._rec = rec
        self._name = name
        self._encrypted = encrypted
        self._t0: Optional[float] = None
        self._rss0: Optional[int] = None
        self._e0_uj: Optional[int] = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._rss0 = _rss_bytes()
        self._e0_uj = _energy_uj()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        t1 = time.perf_counter()
        rss1 = _rss_bytes()
        t0 = self._t0 or t1
        rss0 = self._rss0 or rss1

        e0 = self._e0_uj
        e1 = _energy_uj()
        dt = float(t1 - t0)
        energy_j = 0.0
        power_w = 0.0
        if e0 is not None and e1 is not None and dt > 0.0:
            energy_j = (e1 - e0) / 1e6
            power_w = energy_j / dt

        self._rec.add(
            StepMetric(
                name=self._name,
                seconds=dt,
                rss_delta_bytes=int(rss1 - rss0),
                rss_after_bytes=int(rss1),
                encrypted=self._encrypted,
                energy_joules=energy_j,
                power_watts=power_w,
            )
        )
