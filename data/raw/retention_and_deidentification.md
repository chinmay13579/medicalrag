# Retention and De-identification

## Retention Periods
Clinical documents are kept for the period required by the care provider's national law, which is set per provider contract. The Platform's own operational copies of exchanged messages are deleted after 30 days. Audit logs are retained for 10 years. Support tickets that contain patient identifiers are deleted after 12 months.

## Pseudonymisation versus Anonymisation
Pseudonymised data is still personal data under GDPR because it can be re-identified with the key. Data is called anonymised only after the re-identification risk assessment is signed off by the Data Protection Officer. Free-text clinical notes are never considered anonymised.

## Small Numbers Rule
Any statistic or research extract must have a minimum cell size of 10 patients. Cells with fewer than 10 patients are suppressed or merged into a wider category. Dates of birth are generalised to year, and postal codes to the first three digits.

## Re-identification Prohibition
Attempting to re-identify an individual from de-identified data is prohibited and is treated as a security incident. Researchers sign this prohibition in the data use agreement.

## Medical Report De-identification
Before clinical reports such as laboratory results, radiology reports and discharge summaries are used for search or research, direct identifiers are removed and quasi-identifiers are generalised. Names, national IDs, MRNs, contact details, facility names and accession numbers are replaced with generic markers. Dates of birth are reduced to the year, all other dates to year and month, and ages to five-year bands, and ages above 89 are reported as 90+. The output is labelled de-identified and is treated as anonymised only after Data Protection Officer sign-off. The assistant never links a de-identified report to a named person.
