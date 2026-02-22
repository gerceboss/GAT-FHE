"""Client-side metrics: time, RSS delta, RSS after for keygen, encrypt, decrypt."""

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


@dataclass
class StepMetric:
    name: str
    seconds: float
    rss_delta_bytes: int
    rss_after_bytes: int
    encrypted: bool = True


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
            }
            for m in self._metrics
        }

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
            print(f"  {m.name:<28} {mode:<6} {m.seconds:>8.4f}s  "
                  f"RSS Δ {dmb:>+8.2f} MB  RSS {mb:>8.2f} MB")
        total_time = sum(m.seconds for m in self._metrics)
        total_rss_delta = sum(m.rss_delta_bytes for m in self._metrics) / (1024 * 1024)
        enc_time = sum(m.seconds for m in self._metrics if m.encrypted)
        dec_time = sum(m.seconds for m in self._metrics if not m.encrypted)
        print("-" * 80)
        print(f"  {'TOTAL':<28} {'':6} {total_time:>8.4f}s  "
              f"RSS Δ {total_rss_delta:>+8.2f} MB")
        if total_time > 0:
            print()
            print(f"  Encrypted ops: {enc_time:>8.4f}s ({enc_time / total_time * 100:>5.1f}%)")
            print(f"  Plaintext ops: {dec_time:>8.4f}s ({dec_time / total_time * 100:>5.1f}%)")
        print("=" * 80)


class _StepContext:
    def __init__(self, rec: MetricsRecorder, name: str, encrypted: bool) -> None:
        self._rec = rec
        self._name = name
        self._encrypted = encrypted
        self._t0: Optional[float] = None
        self._rss0: Optional[int] = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._rss0 = _rss_bytes()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        t1 = time.perf_counter()
        rss1 = _rss_bytes()
        t0 = self._t0 or t1
        rss0 = self._rss0 or rss1
        self._rec.add(
            StepMetric(
                name=self._name,
                seconds=float(t1 - t0),
                rss_delta_bytes=int(rss1 - rss0),
                rss_after_bytes=int(rss1),
                encrypted=self._encrypted,
            )
        )
