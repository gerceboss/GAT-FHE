"""
Metrics tracking for FHE GAT encoder.
Tracks time and memory (RSS) for each pipeline step.
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


@dataclass
class StepMetric:
    """Metrics for a single pipeline step."""
    name: str
    seconds: float
    rss_delta_bytes: int
    rss_after_bytes: int
    encrypted: bool = True  # Whether this step was encrypted or plaintext


class MetricsRecorder:
    """
    Records timing and memory metrics for pipeline steps.
    
    Usage:
        metrics = MetricsRecorder()
        
        with metrics.step("linear_layer", encrypted=True):
            # ... perform operation ...
            pass
        
        metrics.print_report()
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
        
        print("\n" + "="*80)
        print("=== Metrics (time + RSS delta) ===")
        print("="*80)
        
        for m in self._metrics:
            mode = "🔒 ENC" if m.encrypted else "🔓 DEC"
            mb = m.rss_after_bytes / (1024 * 1024) if m.rss_after_bytes else 0.0
            dmb = m.rss_delta_bytes / (1024 * 1024)
            print(f"  {m.name:<25} {mode:<8} {m.seconds:>8.4f}s  "
                  f"RSS Δ {dmb:>+8.2f} MB  RSS {mb:>8.2f} MB")
        
        # Print summary
        total_time = sum(m.seconds for m in self._metrics)
        total_rss_delta = sum(m.rss_delta_bytes for m in self._metrics) / (1024 * 1024)
        enc_time = sum(m.seconds for m in self._metrics if m.encrypted)
        dec_time = sum(m.seconds for m in self._metrics if not m.encrypted)
        
        print("-"*80)
        print(f"  {'TOTAL':<25} {'':8} {total_time:>8.4f}s  "
              f"RSS Δ {total_rss_delta:>+8.2f} MB")
        print()
        print(f"  Encrypted ops: {enc_time:>8.4f}s ({enc_time/total_time*100:>5.1f}%)")
        print(f"  Plaintext ops: {dec_time:>8.4f}s ({dec_time/total_time*100:>5.1f}%)")
        print("="*80)


class _StepContext:
    """Context manager for a single step."""
    
    def __init__(self, rec: MetricsRecorder, name: str, encrypted: bool) -> None:
        self._rec = rec
        self._name = name
        self._encrypted = encrypted
        self._t0: Optional[float] = None
        self._rss0: Optional[int] = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._rss0 = rss_bytes()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        t1 = time.perf_counter()
        rss1 = rss_bytes()
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
