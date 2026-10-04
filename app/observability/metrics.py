from __future__ import annotations

import threading
import time
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

REQUESTS = Counter(
    "rag_http_requests_total", "HTTP requests", ("method", "route", "status")
)
REQUEST_LATENCY = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request latency",
    ("method", "route"),
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
RAG_REQUESTS = Counter(
    "rag_requests_total", "RAG requests", ("search_mode", "outcome")
)
RAG_LATENCY = Histogram(
    "rag_request_duration_seconds",
    "End-to-end RAG request latency",
    ("search_mode",),
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
RAG_SOURCES = Histogram(
    "rag_sources_returned", "Number of grounded sources returned", buckets=(0, 1, 2, 3, 5, 10)
)
READINESS = Gauge("rag_dependency_ready", "Dependency readiness", ("dependency",))


def metrics_payload() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST


class SLOWindow:
    """Small in-process rolling counters for a human-readable operational snapshot."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.total = 0
        self.errors = 0
        self.latencies_ms: list[float] = []

    def observe(self, latency_seconds: float, error: bool) -> None:
        with self._lock:
            self.total += 1
            self.errors += int(error)
            self.latencies_ms.append(latency_seconds * 1000)
            if len(self.latencies_ms) > 1000:
                self.latencies_ms = self.latencies_ms[-1000:]

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            values = sorted(self.latencies_ms)
            p95 = values[min(len(values) - 1, int(len(values) * 0.95))] if values else 0.0
            return {
                "window_started_at_epoch": self.started_at,
                "requests": self.total,
                "errors": self.errors,
                "error_rate": self.errors / self.total if self.total else 0.0,
                "p95_latency_ms": round(p95, 2),
            }


slo_window = SLOWindow()
