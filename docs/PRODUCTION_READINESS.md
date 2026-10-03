# Production Readiness

This repository is a reference implementation, not a claim that a local Docker stack is automatically enterprise production-ready.

## Implemented controls

- Typed API contracts and input limits.
- Bounded retrieval attempts.
- Explicit request-level retrieval configuration.
- Structured source lineage and de-duplication.
- Prompt-injection envelope for retrieved evidence.
- Fail-closed pattern for unsafe queries.
- Non-root API container.
- Service health checks.
- CI lint/type/test gates.
- No committed credentials.

## Required before enterprise deployment

- SSO/OIDC authentication and RBAC.
- Tenant/project/ACL filters at retrieval time.
- TLS and authentication for OpenSearch and Redis.
- Secret manager integration.
- Rate limits, quotas and abuse controls.
- Durable tracing/audit retention with PII redaction.
- Real hybrid BM25 + vector search pipeline and embedding service.
- Production LLM adapter with timeout, retry and circuit breaker.
- Cache keys scoped by tenant, project, KB version, model and prompt version.
- Backup/restore, DR, SLOs and alerts.
- SBOM, dependency scanning, container scanning and signed releases.
- Human approval gates for high-impact tools/actions.

## Threats to test

Prompt injection, indirect injection from retrieved documents, data exfiltration, cross-tenant retrieval, poisoned KB content, malicious URLs, oversized input, retry loops, unavailable LLM/search backend, stale cache, citation mismatch and trace leakage.
