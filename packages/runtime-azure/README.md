# UOD RG24 Azure runtime

Synchronous infrastructure ports, deterministic artifact serialization, and
managed-identity Azure Blob Storage and Service Bus adapters for the UOD RG24
preprocessing services.

The package deliberately does **not** resolve logical artifact IDs, reserve or
commit catalogue records, run a Service Bus receiver, or know HTTP/Azure
Functions transports. Those responsibilities remain behind the broker and
service application ports. Blob adapters accept only already-resolved internal
locations and never accept paths from wire commands.

`ArtifactControl` defines the synchronous broker lifecycle as `resolve_input`,
`claim_output`, `commit`, and `mark_failed`; this release intentionally ships
no broker HTTP implementation. `AzureBlobArtifactReader` streams verified
inputs into a private `StepWorkspace`, while `AzureBlobArtifactWriter` uploads
only `data.parquet`, `manifest.json`, and `qc.json` from that workspace. The
Azure clients use `TokenCredential`/`DefaultAzureCredential`; connection-string
construction and live-Azure tests are intentionally absent.

Commit evidence retains the immutable `ArtifactBundleWriteReceipt` for all
three uploaded members plus the primary media type; manifest and QC staging
evidence is not collapsed into the primary data receipt.

The public runtime models include validated W3C trace context, allowlisted safe
log dimensions, SHA-256 and ETag value types, and staged read/write receipts.
Service Bus publishers propagate trace context and canonical JSON, with SDK
failures mapped to stable runtime error codes and redacted `ProblemDetails`.

The first release serializes the five reports produced by sample
harmonization:

| Output | Kind | Schema |
| --- | --- | --- |
| `sampleMap` | `sampleMap` | `sample-map/1.0` |
| `availability` | `sampleAvailability` | `sample-availability/1.0` |
| `retainedCohort` | `retainedCohort` | `retained-cohort/1.0` |
| `dropReport` | `sampleDropReport` | `sample-drop-report/1.0` |
| `alignedSampleIds` | `alignedSampleIds` | `aligned-sample-ids/1.0` |

Each data artifact is deterministic Parquet and receives an
`artifact-qc/1.0` canonical-JSON sidecar. Matrix bytes and source-format
adapters are intentionally deferred.

Run the package tests with Python 3.12:

```bash
uv run --python 3.12 --extra test pytest
```
