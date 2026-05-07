"""
Shared utilities for plain and FHE GAT servers: RSS/energy, metrics CSV, TCP framing.
"""

from __future__ import annotations

import csv
import os
import struct
from typing import Any, Dict, List, Optional, Union

import socket


# ── RSS / energy ─────────────────────────────────────────────────────────────

def rss_bytes() -> int:
    """Return resident set size in bytes (Linux /proc or resource)."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024  # kB → bytes
    except OSError:
        pass
    try:
        import resource
        return int(getattr(resource.getrusage(resource.RUSAGE_SELF), "ru_maxrss", 0)) * 1024
    except Exception:
        return 0
    return 0


def energy_uj() -> Optional[int]:
    """Read energy_uj from /sys/class/powercap if available."""
    try:
        base = "/sys/class/powercap"
        if not os.path.isdir(base):
            return None
        for entry in sorted(os.listdir(base)):
            path = os.path.join(base, entry, "energy_uj")
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    return int(f.read().strip())
    except Exception:
        return None
    return None


# ── Metrics CSV (standard columns: step, server_time, client_time, rss_after_bytes, rss_delta_bytes, power_watts, energy_joules, throughput) ──────────────────────────────────────────────────────────────

METRICS_CSV_FIELDS = (
    "step",
    "server_time",
    "client_time",
    "rss_after_bytes",
    "rss_delta_bytes",
    "power_watts",
    "energy_joules",
    "throughput",
)


def _normalize_metrics_row(
    row: Dict[str, Any],
    step_key: str = "phase",
    time_side: str = "server",
) -> Dict[str, Any]:
    """
    Ensure row has step, server_time, client_time, rss_*, power_watts, energy_joules, throughput.
    time_side: "server" | "client" | "both"
      - server: server_time from seconds/server_time_seconds, client_time=0
      - client: client_time from seconds/client_time, server_time=0
      - both: server_time and client_time from row (server_time_seconds, client_encryption_time+client_decryption_time)
    """
    step = row.get(step_key) or row.get("step") or ""
    rss_after = int(row.get("rss_after_bytes", 0))
    rss_delta = int(row.get("rss_delta_bytes", 0))
    power = float(row.get("power_watts", 0.0))
    energy = float(row.get("energy_joules", 0.0))

    if time_side == "server":
        server_time = float(
            row.get("server_time") or row.get("server_time_seconds") or row.get("seconds", 0.0)
        )
        client_time = 0.0
    elif time_side == "client":
        server_time = 0.0
        client_time = float(row.get("client_time") or row.get("seconds", 0.0))
    else:  # both
        server_time = float(
            row.get("server_time") or row.get("server_time_seconds", 0.0)
        )
        client_time = float(
            row.get("client_time")
            or (float(row.get("client_encryption_time", 0.0)) + float(row.get("client_decryption_time", 0.0)))
        )

    total_time = server_time or client_time
    throughput = (1.0 / total_time) if total_time > 0 else 0.0

    return {
        "step": step,
        "server_time": round(server_time, 6),
        "client_time": round(client_time, 6),
        "rss_after_bytes": rss_after,
        "rss_delta_bytes": rss_delta,
        "power_watts": power,
        "energy_joules": energy,
        "throughput": round(throughput, 6),
    }


def write_metrics_csv(
    path: str,
    data: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
    time_side: str = "server",
    *,
    append: bool = False,
) -> None:
    """
    Write metrics to a CSV file with standard columns:
    step, server_time, client_time, rss_after_bytes, rss_delta_bytes, power_watts, energy_joules, throughput.

    time_side: "server" (default) | "client" | "both"
      Use "server" for server-side metrics, "client" for client-side (keygen, encrypt, decrypt phases), "both" for per-batch rows that have both.
    append: when True, append rows to an existing file (write header only if missing/empty).
    """
    if not data:
        return

    if isinstance(data, list):
        rows = [_normalize_metrics_row(r, time_side=time_side) for r in data]
    else:
        rows = []
        for name, m in data.items():
            row = dict(m)
            row["phase"] = name
            rows.append(_normalize_metrics_row(row, step_key="phase", time_side=time_side))

    if not rows:
        return

    file_existed = append and os.path.isfile(path) and os.path.getsize(path) > 0
    mode = "a" if append else "w"
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(METRICS_CSV_FIELDS), extrasaction="ignore")
        if not file_existed:
            writer.writeheader()
        writer.writerows(rows)

    if not append:
        print(f"[server] metrics written → {path}")


def append_metrics_rows(
    path: str,
    data: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
    time_side: str = "server",
) -> None:
    """Convenience wrapper: append normalised rows; create file with header if needed."""
    write_metrics_csv(path, data, time_side=time_side, append=True)


def append_dict_rows(
    path: str,
    rows: List[Dict[str, Any]],
    fieldnames: List[str],
) -> None:
    """
    Append arbitrary-schema rows to a CSV (write header if file is new/empty).
    Used for client-side per-batch metrics where columns differ from the standard schema.
    """
    if not rows:
        return
    file_existed = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_existed:
            writer.writeheader()
        writer.writerows(rows)


# ── TCP framing ──────────────────────────────────────────────────────────────

def recvall(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from socket."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed mid-receive")
        buf.extend(chunk)
    return bytes(buf)


def send_frame(sock: socket.socket, data: bytes) -> None:
    """Send 4-byte big-endian length + payload."""
    sock.sendall(struct.pack(">I", len(data)) + data)


def recv_frame(sock: socket.socket) -> bytes:
    """Read 4-byte length then payload."""
    raw_len = recvall(sock, 4)
    length = struct.unpack(">I", raw_len)[0]
    return recvall(sock, length)


def send_ok(sock: socket.socket) -> None:
    """Send status byte 0 (success)."""
    sock.sendall(b"\x00")


def send_error(sock: socket.socket, exc: Exception) -> None:
    """Send status byte 1 + 4-byte length + error message."""
    msg = str(exc).encode()
    sock.sendall(b"\x01" + struct.pack(">I", len(msg)) + msg)
