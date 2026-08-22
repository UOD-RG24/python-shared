# Versioned shared distributions

The V1 architecture is introduced beside the legacy `uod-rg24-python-shared`
package so deployed Function Apps can migrate without an in-place break.

- `contracts`: strict wire models and generated JSON Schemas.
- `preprocessing`: pure, Azure-free scientific strategies.

The planned `runtime-azure` distribution will be added only when the artifact
broker interfaces and managed-identity configuration are implemented. Existing
modules under `src/uod_rg24` remain the V0 compatibility surface.

