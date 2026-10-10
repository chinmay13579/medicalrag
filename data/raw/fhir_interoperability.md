# FHIR Interoperability Standards

## Supported Formats
The Platform exposes and consumes HL7 FHIR R4 resources. Legacy hospital systems that send HL7 v2 messages (ADT, ORU, ORM) are converted to FHIR by the ingestion gateway. CDA documents are accepted for Czech and Slovak ePrescription exchange.

## Terminology Mapping
Diagnoses are coded in ICD-10, laboratory observations in LOINC, medications in ATC with local product codes mapped to it, and clinical findings in SNOMED CT where a licence is available. Unmapped local codes are never dropped; they are stored as text with the flag needs-mapping and reviewed by the terminology team every week.

## Validation and Rejected Messages
Every incoming resource is validated against the FHIR R4 base profile and the relevant national profile. A resource that fails validation is quarantined for 14 days and the sending system receives an OperationOutcome that lists each error. Quarantined messages are never shown to clinicians.

## Patient Summary
The International Patient Summary (IPS) is the default document for cross-border exchange. It contains allergies, current medications, problems, immunisations and recent procedures, and it is generated on demand rather than stored.
