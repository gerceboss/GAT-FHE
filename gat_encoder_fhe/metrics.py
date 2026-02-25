"""
Metrics tracking for FHE GAT encoder.
Tracks time, memory (RSS), and optional energy/power for each pipeline step.
"""

import time
from dataclasses import dataclass
from typing import Optional


def _read_proc_status_rss_bytes() -> Optional[int]:
    """
    Return current RSS in bytes using /proc/self/status if available (Linux).
    """
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # e.g. "VmRSS:\t  123456 kB"
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except OSError:
        return None
    return None


def rss_bytes() -> int:
    """
    Best-effort resident set size (RSS) in bytes.
    """
    rss = _read_proc_status_rss_bytes()
    if rss is not None:
        return rss

    # Fallback (may be peak RSS depending on platform)
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF)
        # On Linux ru_maxrss is in KB.
        return int(ru.ru_maxrss) * 1024
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
    """Metrics for a single pipeline step."""

    name: str
    seconds: float
    rss_delta_bytes: int
    rss_after_bytes: int
    encrypted: bool = True  # Whether this step was encrypted or plaintext
    energy_joules: float = 0.0
    power_watts: float = 0.0


class MetricsRecorder:
    """
    Records timing, memory, and optional energy/power metrics for pipeline steps.
    """

    def __init__(self) -> None:
        self._metrics: list[StepMetric] = []

    def step(self, name: str, encrypted: bool = True):
        """Context manager for tracking a single step."""
        return _StepContext(self, name, encrypted)

    def add(self, metric: StepMetric) -> None:
        """Add a metric."""
        self._metrics.append(metric)

    @property
    def metrics(self) -> list[StepMetric]:
        """Get all recorded metrics."""
        return list(self._metrics)

    def print_report(self) -> None:
        """Print formatted metrics report."""
        if not self._metrics:
            return

        print("\n" + "=" * 80)
        print("=== Metrics (time + RSS delta) ===")
        print("=" * 80)

        for m in self._metrics:
            mode = "🔒 ENC" if m.encrypted else "🔓 DEC"
            mb = m.rss_after_bytes / (1024 * 1024) if m.rss_after_bytes else 0.0
            dmb = m.rss_delta_bytes / (1024 * 1024)
            print(
                f"  {m.name:<25} {mode:<8} {m.seconds:>8.4f}s  "
                f"RSS Δ {dmb:>+8.2f} MB  RSS {mb:>8.2f} MB"
            )

        # Print summary
        total_time = sum(m.seconds for m in self._metrics)
        total_rss_delta = sum(m.rss_delta_bytes for m in self._metrics) / (1024 * 1024)
        enc_time = sum(m.seconds for m in self._metrics if m.encrypted)
        dec_time = sum(m.seconds for m in self._metrics if not m.encrypted)

        print("-" * 80)
        print(
            f"  {'TOTAL':<25} {'':8} {total_time:>8.4f}s  "
            f"RSS Δ {total_rss_delta:>+8.2f} MB"
        )
        print()
        print(f"  Encrypted ops: {enc_time:>8.4f}s ({enc_time/total_time*100:>5.1f}%)")
        print(f"  Plaintext ops: {dec_time:>8.4f}s ({dec_time/total_time*100:>5.1f}%)")
        print("=" * 80)

    def to_dict(self) -> dict[str, any]:
        """Convert metrics to a dictionary."""
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

    def write_txt(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "name,seconds,rss_delta_bytes,rss_after_bytes,encrypted,energy_joules,power_watts\n"
            )
            for m in self._metrics:
                f.write(
                    f"{m.name},{m.seconds:.6f},{m.rss_delta_bytes},{m.rss_after_bytes},"
                    f"{int(m.encrypted)},{m.energy_joules:.6f},{m.power_watts:.6f}\n"
                )


class _StepContext:
    """Context manager for a single step."""

    def __init__(self, rec: MetricsRecorder, name: str, encrypted: bool) -> None:
        self._rec = rec
        self._name = name
        self._encrypted = encrypted
        self._t0: Optional[float] = None
        self._rss0: Optional[int] = None
        self._e0_uj: Optional[int] = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._rss0 = rss_bytes()
        self._e0_uj = _energy_uj()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        t1 = time.perf_counter()
        rss1 = rss_bytes()
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
