# Versioned shared distributions

The V1 architecture is introduced beside the legacy `uod-rg24-python-shared`
package so deployed Function Apps can migrate without an in-place break.

- `contracts`: strict wire models and generated JSON Schemas.
- `preprocessing`: pure, Azure-free scientific strategies.
- `runtime-azure`: artifact runtime ports, deterministic serializers, and
  managed-identity Azure Blob/Service Bus adapters.

The three V1 distributions share the lock and quality-gate configuration in
this directory while remaining independently buildable and versioned. Existing
modules under `src/uod_rg24` remain the V0 compatibility surface.
