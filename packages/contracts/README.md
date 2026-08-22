# UOD RG24 contracts

Strict, versioned wire models and generated JSON Schemas shared by the genome
platform control plane, jobs coordinator, artifact broker, and preprocessing
services.

Generate the checked-in schemas with:

```bash
uv run --extra test uod-rg24-export-schemas
```

Run the contract tests with:

```bash
uv run --extra test pytest
```

