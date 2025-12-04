import pandas as pd
import numpy as np
import os

np.random.seed(42)

rows = 3000   # number of fake transactions
data = []

for i in range(rows):
    Time = np.random.randint(0, 100000)

    V1 = np.random.normal()
    V2 = np.random.normal()
    V3 = np.random.normal()
    V4 = np.random.normal()
    V5 = np.random.normal()

    Amount = round(np.random.exponential(scale=300), 2)

    # Only 3% fraud
    if np.random.rand() < 0.03 and Amount > 800:
        Class = 1   # Fraud
    else:
        Class = 0   # Normal

    data.append([Time, V1, V2, V3, V4, V5, Amount, Class])

columns = ["Time","V1","V2","V3","V4","V5","Amount","Class"]
df = pd.DataFrame(data, columns=columns)

os.makedirs("data", exist_ok=True)
df.to_csv("data/creditcard.csv", index=False)

print("✅ Fake fraud dataset created successfully!")
