# Orders schema and lineage (synthetic demonstration)
Owner: Analytics Engineering. This is sample documentation, not a real customer system.

The orders table is built from raw_checkout_events, joined to customer_profiles on customer_id,
then transformed by the daily orders_normalization job into analytics.orders.
Downstream consumers include revenue_daily and delivery_performance.

order_id is a unique business identifier, not a meaningful numeric model feature.
amount is the order amount in major currency units, such as dollars; never mix cents with dollars.
delivery_days is elapsed calendar days from dispatch to delivery.
customer_region comes from the customer_profiles join. Missing customer_region values can indicate
an unmatched customer key, a delayed dimension refresh, or an optional source field.
CSV inferred types are not a substitute for a formal schema contract.
