# Access Control and Audit

## Role-Based Access
Clinicians see only records of patients they are treating or are explicitly assigned to. Support engineers have no access to clinical content by default and see only technical metadata such as message ids and error codes. Data stewards may view pseudonymised data only.

## Break-Glass Access
In an emergency a clinician may use break-glass access to open a record outside their assignment. The clinician must enter a reason, the patient and the care provider's privacy officer are notified immediately, and every break-glass event is reviewed within 5 business days.

## Audit Logging
Every read of a patient record is logged with user, patient pseudonym, purpose and timestamp. Patients can request their access log through the portal and receive it within 30 days. Logs are append-only and cannot be edited by administrators.

## Access Reviews
Privileged access is reviewed every quarter and expires automatically after 90 days unless a manager renews it. Accounts of staff who leave are disabled within 24 hours.
