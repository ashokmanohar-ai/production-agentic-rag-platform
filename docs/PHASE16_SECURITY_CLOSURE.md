# Phase 16 Enterprise Security Closure

Phase 16 closes the highest-risk authorization gaps that remained after the security foundation.

API-key authentication no longer accepts caller-selected X-Role as authorization. Both OIDC and API-key identities require an active persisted membership for the selected tenant and project, and the membership role is authoritative.

Evaluation runs now persist tenant and project scope. Evaluation history and the quality dashboard filter by the authenticated scope. Detail and comparison endpoints verify that requested run IDs belong to the same authenticated tenant/project before returning data.

The unsafe in-memory upload and list endpoints are retired with HTTP 410 after authentication and role checks. The legacy single-document lookup is also disabled with HTTP 410. Durable document APIs remain the supported knowledge-management surface.

A schema migration adds evaluation tenant/project columns and a trace-ownership table for the next feedback-binding step.

## Remaining boundary

Trace ownership persistence is introduced at the schema/repository layer, but feedback is not yet bound to that ownership record in the API path. Feedback remains role-protected. Audit coverage is also still incomplete for retry, reindex, delete, evaluation and feedback actions. These items should be completed before calling authorization and audit closure fully complete.
