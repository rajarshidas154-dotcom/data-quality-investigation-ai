# Runbook: amount spikes and currency units
Applies to analytics.orders and the amount column.
If amount jumps by approximately 100 times, investigate a cents-to-major-unit conversion change.
This is a hypothesis to check, not proof of a production defect.
Compare affected records with raw_checkout_events and inspect the orders_normalization deployment.
Check currency codes and conversion rules before comparing values across currencies.
Segment by source and ingestion date. Confirm whether a business promotion explains the change.
Do not divide values automatically: obtain the owner-approved correction after confirming the cause.
