"""Regenerates the built-in example datasets in app/samples (synthetic, no real people)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

OUT = Path(__file__).resolve().parent.parent / "app" / "samples"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(7)

# Customer churn -------------------------------------------------------------
n = 800
months = rng.integers(1, 72, n)
plan = rng.choice(["Basic", "Standard", "Premium"], n, p=[0.45, 0.35, 0.2])
contract = rng.choice(["Monthly", "Yearly"], n, p=[0.6, 0.4])
bill = np.round(np.select([plan == "Basic", plan == "Standard"], [20, 45], 80) + rng.normal(0, 6, n), 2)
calls = rng.poisson(1.5, n)
score = -1.6 - 0.09 * months + 1.1 * calls + 2.6 * (contract == "Monthly") + 0.02 * bill
churn = rng.random(n) < 1 / (1 + np.exp(-score))
df = pd.DataFrame(
    {
        "customer_id": [f"C{10000 + i}" for i in range(n)],
        "months_as_customer": months,
        "plan": plan,
        "contract": contract,
        "monthly_bill": bill,
        "support_calls": calls.astype(float),
        "city": rng.choice(["Chennai", "Mumbai", "Delhi", "Bengaluru", "Pune"], n),
        "cancelled": np.where(churn, "Yes", "No"),
    }
)
df.loc[rng.choice(n, 30, replace=False), "support_calls"] = np.nan  # a few realistic blanks
df.to_csv(OUT / "customer-churn.csv", index=False)

# House prices ---------------------------------------------------------------
n = 700
area = rng.integers(450, 3200, n)
beds = np.clip((area / 650 + rng.normal(0, 0.7, n)).round(), 1, 6).astype(int)
age = rng.integers(0, 45, n)
hood = rng.choice(["Suburb", "City centre", "Riverside", "Outskirts"], n, p=[0.4, 0.25, 0.15, 0.2])
premium = pd.Series(hood).map({"Suburb": 1.0, "City centre": 1.45, "Riverside": 1.3, "Outskirts": 0.75}).to_numpy()
parking = rng.choice(["Yes", "No"], n, p=[0.65, 0.35])
price = (area * 95 * premium + beds * 9000 - age * 1400 + (parking == "Yes") * 12000) * rng.normal(1, 0.08, n)
pd.DataFrame(
    {
        "area_sqft": area,
        "bedrooms": beds,
        "age_years": age,
        "neighbourhood": hood,
        "parking": parking,
        "price": price.round(-2).astype(int),
    }
).to_csv(OUT / "house-prices.csv", index=False)

# Iris flowers ---------------------------------------------------------------
iris = load_iris(as_frame=True)
flowers = iris.frame.rename(columns=lambda c: c.replace(" (cm)", "_cm").replace(" ", "_"))
flowers["species"] = flowers.pop("target").map(dict(enumerate(iris.target_names)))
flowers.to_csv(OUT / "iris-flowers.csv", index=False)
print("wrote", *(p.name for p in sorted(OUT.glob("*.csv"))))
