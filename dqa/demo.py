"""Generate reproducible synthetic examples. Never overwrite existing datasets."""
import numpy as np
import pandas as pd
from dqa.core import ROOT


def main():
    rng = np.random.default_rng(42)
    folder = ROOT / "data"
    folder.mkdir(parents=True, exist_ok=True)
    baseline = pd.DataFrame({"order_id": np.arange(1, 501),
        "amount": rng.normal(120, 18, 500).round(2),
        "delivery_days": rng.integers(1, 6, 500),
        "customer_region": rng.choice(["North", "South", "East", "West"], 500)})
    current = baseline.copy()
    current["order_id"] += 500
    current.loc[0:19, "amount"] *= 100
    current.loc[20:59, "customer_region"] = None
    current.loc[60:74, "delivery_days"] = 40
    current = pd.concat([current, current.iloc[100:110]], ignore_index=True)
    for name, frame in [("orders_baseline.csv", baseline), ("orders_current.csv", current)]:
        target = folder / name
        if not target.exists():
            frame.to_csv(target, index=False)
            print(f"Created {target}")
        else:
            print(f"Keeping existing {target}")


if __name__ == "__main__":
    main()
