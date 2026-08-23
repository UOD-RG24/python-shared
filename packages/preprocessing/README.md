# UOD RG24 preprocessing

Transport-independent scientific preprocessing strategies. The first strategy
implements deterministic TCGA sample harmonization without Azure, HTTP, or
filesystem dependencies.

```bash
uv run --extra test pytest
```

The policy deliberately records patient-level fallback when a source sample is
absent from the clinical sample table but its parsed patient exists in the
clinical patient table. Contradictory clinical mappings remain terminal.
