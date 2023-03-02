import pandas as pd
import os

df = pd.read_csv("3DSC_MP.csv", skiprows=1, usecols=["cif","tc", "formula_sc"])[["cif","tc","formula_sc"]]
print(df)
df["id"] = df.apply(lambda x: os.path.basename(x["cif"]).rsplit('.',1)[0], axis=1)
print(df)
df = df.rename(columns={"formula_sc": "pretty_formula"})
df.to_csv("database/targets_SC_v2.csv", columns=['id', 'tc','pretty_formula'], index=False)