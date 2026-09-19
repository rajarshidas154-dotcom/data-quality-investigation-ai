# Runbook: missing regions and duplicate orders
For missing customer_region, check unmatched customer keys against customer_profiles.
Inspect dimension refresh completion and left-join coverage before blaming ingestion.
Compare missingness as a percentage of rows, not only absolute counts.

For duplicate rows, inspect retry and replay behavior in raw_checkout_events.
Full-row duplicates differ from duplicate business keys: also check order_id uniqueness separately.
Do not remove rows automatically. Confirm replay behavior and downstream consumers with the data owner.

For elevated delivery_days, check carrier feed delays, timezone conversions and actual disruptions.
A statistical outlier can be a valid late delivery. Check service-level rules before labeling it an error.
