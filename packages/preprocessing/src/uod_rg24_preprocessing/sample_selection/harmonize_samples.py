from __future__ import annotations

from collections import defaultdict
from hashlib import sha256

from .errors import (
    ClinicalMappingConflictError,
    DuplicateSampleCandidateError,
    HarmonizationError,
    NoCommonSamplesError,
)
from .models import (
    SIDECAR_ROLES,
    AvailabilityRow,
    CohortPolicy,
    DropReportRow,
    HarmonizationQC,
    HarmonizationRequest,
    HarmonizationResult,
    MatchLevel,
    SampleCandidate,
    SampleMapRow,
)
from .policies.tcga_primary_tumour_v1 import ValidatedCandidate, select_replicate
from .tcga import TcgaBarcode, parse_tcga_barcode


def _modality_for_role(role: str) -> str:
    if role.startswith("clinical"):
        return "clinical"
    return role


def _validate_clinical_map(request: HarmonizationRequest) -> frozenset[str]:
    """Return the clinical sample keys, in both full-barcode and canonical form.

    Every value in ``clinicalSampleToPatient`` must agree with the patient
    encoded in its own key, so once the mapping is validated it carries no
    information beyond membership.
    """
    patient_ids = {item.strip().upper() for item in request.clinical_patient_ids}
    known_samples: set[str] = set()
    for sample_id, mapped_patient_id in request.clinical_sample_to_patient.items():
        barcode = parse_tcga_barcode(sample_id)
        if not barcode.is_sample:
            raise ClinicalMappingConflictError(
                f"Clinical sample key is not a sample barcode: {sample_id}"
            )
        patient_id = mapped_patient_id.strip().upper()
        if barcode.patient_id != patient_id:
            raise ClinicalMappingConflictError(
                f"Clinical mapping for {sample_id} declares {patient_id}, "
                f"but the barcode belongs to {barcode.patient_id}."
            )
        if patient_ids and patient_id not in patient_ids:
            raise ClinicalMappingConflictError(
                f"Clinical sample {sample_id} refers to absent patient {patient_id}."
            )
        known_samples.add(barcode.normalized)
        known_samples.add(barcode.canonical_sample_id or barcode.normalized)
    return frozenset(known_samples)


def _candidate_barcode(candidate: SampleCandidate) -> TcgaBarcode:
    value = candidate.source_sample_id or candidate.patient_id
    if value is None:
        raise HarmonizationError(
            f"Candidate for role {candidate.role} declares neither "
            "sourceSampleId nor patientId."
        )
    barcode = parse_tcga_barcode(value)
    if candidate.patient_id is not None:
        declared_patient = candidate.patient_id.strip().upper()
        if barcode.patient_id != declared_patient:
            raise ClinicalMappingConflictError(
                f"Candidate {value} declares patient {declared_patient}, "
                f"but its barcode belongs to {barcode.patient_id}."
            )
    return barcode


def _mapping_status(
    barcode: TcgaBarcode,
    clinical_samples: frozenset[str],
    clinical_patients: set[str],
) -> tuple[str, str | None]:
    if not clinical_samples and not clinical_patients:
        return "barcodeDerived", None
    if not barcode.is_sample:
        if clinical_patients and barcode.patient_id not in clinical_patients:
            return "clinicalPatientMissing", "missingClinicalPatient"
        return "patientRecord", None

    if barcode.normalized in clinical_samples or (
        barcode.canonical_sample_id is not None
        and barcode.canonical_sample_id in clinical_samples
    ):
        return "clinicalSampleValidated", None
    if barcode.patient_id in clinical_patients:
        return "patientLevelFallback", None
    return "clinicalPatientMissing", "missingClinicalPatient"


def _sample_order_hash(patient_ids: list[str]) -> str:
    payload = "".join(f"{patient_id}\n" for patient_id in patient_ids)
    return sha256(payload.encode("utf-8")).hexdigest()


def harmonize_samples(request: HarmonizationRequest) -> HarmonizationResult:
    sidecars = SIDECAR_ROLES
    ignored_sidecars = sorted(
        {
            candidate.role
            for candidate in request.candidates
            if candidate.role in sidecars
        }
    )
    active_candidates = [
        candidate for candidate in request.candidates if candidate.role not in sidecars
    ]
    roles = sorted(request.cohort_roles or {item.role for item in active_candidates})
    if len(roles) < 2:
        raise ValueError("harmonizeSamples requires at least two cohort-bearing roles")

    clinical_patients = {
        patient_id.strip().upper() for patient_id in request.clinical_patient_ids
    }
    clinical_samples = _validate_clinical_map(request)

    role_set = set(roles)
    validated: list[ValidatedCandidate] = []
    sample_map: list[SampleMapRow] = []
    drop_report: list[DropReportRow] = []
    identities: set[tuple[str, str]] = set()
    observed_pairs: set[tuple[str, str]] = set()
    fallback_count = 0
    barcode_derived_count = 0
    unused_role_count = 0

    for candidate in active_candidates:
        barcode = _candidate_barcode(candidate)
        if candidate.role not in role_set:
            # Recorded rather than skipped: a candidate the caller supplied for
            # a role outside cohortRoles must still be accounted for.
            unused_role_count += 1
            sample_map.append(
                SampleMapRow(
                    entity_id=barcode.patient_id,
                    patient_id=barcode.patient_id,
                    modality=_modality_for_role(candidate.role),
                    role=candidate.role,
                    source_sample_id=candidate.source_sample_id,
                    selected=False,
                    selection_reason=None,
                    rejected_reason="roleNotInCohort",
                    replicate_group=f"{barcode.patient_id}|{candidate.role}",
                    source_artifact_id=candidate.source_artifact_id,
                    mapping_status="notEvaluated",
                )
            )
            drop_report.append(
                DropReportRow(
                    patient_id=barcode.patient_id,
                    role=candidate.role,
                    reason="roleNotInCohort",
                    source_sample_id=candidate.source_sample_id,
                )
            )
            continue
        identity = (candidate.role, barcode.normalized)
        if identity in identities:
            raise DuplicateSampleCandidateError(
                f"Duplicate sample {barcode.normalized} for role {candidate.role}."
            )
        identities.add(identity)
        observed_pairs.add((barcode.patient_id, candidate.role))
        mapping_status, rejection = _mapping_status(
            barcode, clinical_samples, clinical_patients
        )
        if mapping_status == "patientLevelFallback":
            fallback_count += 1
        elif mapping_status == "barcodeDerived":
            barcode_derived_count += 1
        if rejection is not None:
            sample_map.append(
                SampleMapRow(
                    entity_id=barcode.patient_id,
                    patient_id=barcode.patient_id,
                    modality=_modality_for_role(candidate.role),
                    role=candidate.role,
                    source_sample_id=candidate.source_sample_id,
                    selected=False,
                    selection_reason=None,
                    rejected_reason=rejection,
                    replicate_group=f"{barcode.patient_id}|{candidate.role}",
                    source_artifact_id=candidate.source_artifact_id,
                    mapping_status=mapping_status,
                )
            )
            drop_report.append(
                DropReportRow(
                    patient_id=barcode.patient_id,
                    role=candidate.role,
                    reason=rejection,
                    source_sample_id=candidate.source_sample_id,
                )
            )
            continue
        validated.append(
            ValidatedCandidate(
                candidate=candidate,
                barcode=barcode,
                mapping_status=mapping_status,
            )
        )

    grouped: dict[tuple[str, str], list[ValidatedCandidate]] = defaultdict(list)
    for item in validated:
        grouped[(item.barcode.patient_id, item.candidate.role)].append(item)

    sample_roles = sorted(
        {
            item.candidate.role
            for item in validated
            if item.candidate.role in role_set and item.barcode.is_sample
        }
    )
    exact_samples_by_patient: dict[str, set[str]] = {}
    if request.match_level == MatchLevel.EXACT_SAMPLE:
        validated_patients = {item.barcode.patient_id for item in validated}
        for patient_id in validated_patients:
            role_sample_ids: list[set[str]] = [
                {
                    item.barcode.canonical_sample_id
                    for item in grouped.get((patient_id, role), [])
                    if item.barcode.canonical_sample_id is not None
                }
                for role in sample_roles
            ]
            exact_samples_by_patient[patient_id] = (
                role_sample_ids[0].intersection(*role_sample_ids[1:])
                if len(role_sample_ids) >= 2 and all(role_sample_ids)
                else set()
            )

    selected_by_patient_role: dict[tuple[str, str], ValidatedCandidate] = {}
    for patient_id, role in sorted(grouped):
        group_candidates = grouped[(patient_id, role)]
        if request.match_level == MatchLevel.EXACT_SAMPLE and role in sample_roles:
            exact_samples = exact_samples_by_patient.get(patient_id, set())
            eligible_candidates = [
                item
                for item in group_candidates
                if item.barcode.canonical_sample_id in exact_samples
            ]
            eligible_ids = {id(item) for item in eligible_candidates}
            for item in group_candidates:
                if id(item) in eligible_ids:
                    continue
                sample_map.append(
                    SampleMapRow(
                        entity_id=patient_id,
                        patient_id=patient_id,
                        modality=_modality_for_role(role),
                        role=role,
                        source_sample_id=item.candidate.source_sample_id,
                        selected=False,
                        selection_reason=None,
                        rejected_reason="notInExactSampleIntersection",
                        replicate_group=f"{patient_id}|{role}",
                        source_artifact_id=item.candidate.source_artifact_id,
                        mapping_status=item.mapping_status,
                    )
                )
                drop_report.append(
                    DropReportRow(
                        patient_id=patient_id,
                        role=role,
                        reason="notInExactSampleIntersection",
                        source_sample_id=item.candidate.source_sample_id,
                    )
                )
            if not eligible_candidates:
                continue
        else:
            eligible_candidates = group_candidates

        selected, selection_reason, replicate_rejections = select_replicate(
            eligible_candidates
        )
        selected_by_patient_role[(patient_id, role)] = selected
        for item in sorted(
            eligible_candidates,
            key=lambda entry: (
                entry.barcode.normalized,
                entry.candidate.source_artifact_id,
            ),
        ):
            candidate_identity = (
                item.candidate.source_artifact_id,
                item.candidate.source_sample_id or item.barcode.patient_id,
            )
            is_selected = item is selected
            rejected_reason = (
                None if is_selected else replicate_rejections[candidate_identity]
            )
            sample_map.append(
                SampleMapRow(
                    entity_id=patient_id,
                    patient_id=patient_id,
                    modality=_modality_for_role(role),
                    role=role,
                    source_sample_id=item.candidate.source_sample_id,
                    selected=is_selected,
                    selection_reason=selection_reason if is_selected else None,
                    rejected_reason=rejected_reason,
                    replicate_group=f"{patient_id}|{role}",
                    source_artifact_id=item.candidate.source_artifact_id,
                    mapping_status=item.mapping_status,
                )
            )
            if rejected_reason is not None:
                drop_report.append(
                    DropReportRow(
                        patient_id=patient_id,
                        role=role,
                        reason=rejected_reason,
                        source_sample_id=item.candidate.source_sample_id,
                    )
                )

    patients = sorted({patient_id for patient_id, _ in observed_pairs})
    availability: list[AvailabilityRow] = []
    retained: list[str] = []
    for patient_id in patients:
        role_availability = {
            role: (patient_id, role) in selected_by_patient_role for role in roles
        }
        selected_candidates = {
            role: selected_by_patient_role.get((patient_id, role)) for role in roles
        }
        # aligned_sample_ids and selected_samples both report the source sample
        # ID so the two outputs can be joined; the canonical ID is only used
        # internally to decide exact-sample overlap.
        selected_samples = {
            role: (None if item is None else item.candidate.source_sample_id)
            for role, item in selected_candidates.items()
        }
        canonical_samples = [
            item.barcode.canonical_sample_id
            for item in selected_candidates.values()
            if item is not None and item.barcode.canonical_sample_id is not None
        ]
        exact_overlap = len(canonical_samples) >= 2 and len(set(canonical_samples)) == 1
        keep = (
            any(role_availability.values())
            if request.cohort_policy == CohortPolicy.AVAILABLE
            else all(role_availability.values())
        )
        if request.match_level == MatchLevel.EXACT_SAMPLE:
            keep = keep and exact_overlap
        availability.append(
            AvailabilityRow(
                entity_id=patient_id,
                patient_id=patient_id,
                role_availability=role_availability,
                selected_samples=selected_samples,
                exact_sample_overlap=exact_overlap,
                retained=keep,
            )
        )
        if keep:
            retained.append(patient_id)
            continue
        for role, present in role_availability.items():
            # A role whose candidates were rejected earlier already carries its
            # own drop reason; only a role with no candidates at all is missing.
            if present or (patient_id, role) in observed_pairs:
                continue
            drop_report.append(
                DropReportRow(
                    patient_id=patient_id,
                    role=role,
                    reason="missingRequiredRole",
                )
            )
        if request.match_level == MatchLevel.EXACT_SAMPLE and not exact_overlap:
            drop_report.append(
                DropReportRow(
                    patient_id=patient_id,
                    role="cohort",
                    reason="noExactSampleOverlap",
                )
            )

    if request.cohort_policy == CohortPolicy.INTERSECTION and not retained:
        raise NoCommonSamplesError(
            "No patients satisfy the requested cohort intersection."
        )

    retained_set = set(retained)
    for index, row in enumerate(sample_map):
        if row.selected and row.patient_id not in retained_set:
            sample_map[index] = row.model_copy(
                update={
                    "selected": False,
                    "selection_reason": None,
                    "rejected_reason": "notInRetainedCohort",
                }
            )
            drop_report.append(
                DropReportRow(
                    patient_id=row.patient_id,
                    role=row.role,
                    reason="notInRetainedCohort",
                    source_sample_id=row.source_sample_id,
                )
            )

    sample_map.sort(
        key=lambda row: (
            row.patient_id,
            row.role,
            row.source_sample_id or "",
            row.source_artifact_id,
        )
    )
    drop_report.sort(
        key=lambda row: (
            row.patient_id,
            row.role,
            row.reason,
            row.source_sample_id or "",
        )
    )
    aligned_sample_ids = {
        role: [
            (
                selected_by_patient_role[(patient_id, role)].candidate.source_sample_id
                if (patient_id, role) in selected_by_patient_role
                else None
            )
            for patient_id in retained
        ]
        for role in roles
    }
    selected_count = sum(row.selected for row in sample_map)
    warnings: list[str] = []
    if fallback_count > 0:
        warnings.append(f"patientLevelFallback:{fallback_count}")
    if barcode_derived_count > 0:
        warnings.append(f"clinicalValidationUnavailable:{barcode_derived_count}")
    if unused_role_count > 0:
        warnings.append(f"roleNotInCohort:{unused_role_count}")
    return HarmonizationResult(
        sample_map=sample_map,
        availability=availability,
        retained_cohort=retained,
        drop_report=drop_report,
        aligned_sample_ids=aligned_sample_ids,
        sample_order_sha256=_sample_order_hash(retained),
        qc=HarmonizationQC(
            candidate_count=len(active_candidates),
            valid_candidate_count=len(validated),
            selected_candidate_count=selected_count,
            rejected_candidate_count=len(sample_map) - selected_count,
            available_patient_count=len(patients),
            retained_patient_count=len(retained),
            dropped_patient_count=len(patients) - len(retained),
            patient_level_fallback_count=fallback_count,
            barcode_derived_count=barcode_derived_count,
            cohort_roles=roles,
            ignored_sidecar_roles=ignored_sidecars,
            warnings=warnings,
        ),
    )
