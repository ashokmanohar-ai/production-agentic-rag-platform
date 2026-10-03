# Security Policy

Do not report secrets in public issues. Rotate any credential that is accidentally committed.

The local Compose configuration intentionally disables OpenSearch security for developer convenience and must not be exposed publicly. Production deployments require authentication, TLS, private networking, least-privilege identities, retrieval-time authorization and centralized secret management.

Agent tools must not independently install software, download arbitrary external files, change infrastructure, or perform privileged actions without an explicit authorization and approval policy.
