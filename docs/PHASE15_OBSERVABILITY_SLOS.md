# Phase 15 Observability, SLOs and Reliability

Phase 15 adds a vendor-neutral operational telemetry foundation.

## Metrics

The /metrics endpoint exposes Prometheus text metrics for HTTP request counts and latency, RAG outcomes and latency, returned source counts, and dependency readiness. Route templates are used as labels so document IDs and other high-cardinality path values are not emitted.

## Correlation

Every HTTP request receives an X-Correlation-ID response header. A caller-provided value is propagated with a bounded length; otherwise the platform creates one. Application JSON logs include the active correlation ID.

## SLO snapshot

Administrators can inspect /api/v1/operations/slo. The current in-process window reports RAG request volume, server-error rate, and p95 latency. This is an operational convenience, not a durable monitoring store. Production SLO calculations should come from the exported metrics in a monitoring backend.

Suggested initial objectives:

- Availability: 99.9% successful non-5xx RAG requests over 30 days.
- Latency: 95% of RAG requests complete within 5 seconds, adjusted after production baselining.
- Dependency readiness: required dependencies should remain ready during serving periods.
- Grounding proxy: monitor source-count distribution and evaluation-gate results; source count alone is not a correctness metric.

## Reliability boundaries

Ollama HTTP calls remain bounded by a configurable timeout. Automatic retries are intentionally not added to generation requests because retries can duplicate expensive inference and hide overload. Retry policy should be added only with explicit idempotency, retryable-error classification, backoff, and an overall request deadline.

The in-process SLO window resets on restart and is not suitable as a compliance record. Prometheus-compatible metrics are the durable integration surface for external monitoring and alerting.
