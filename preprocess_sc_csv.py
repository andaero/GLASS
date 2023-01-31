import pandas as pd
import os

df = pd.read_csv("3DSC_MP.csv", skiprows=1, usecols=["cif","tc"])[["cif","tc"]]
print(df)
df["id"] = df.apply(lambda x: os.path.basename(x["cif"]).rsplit('.',1)[0], axis=1)
print(df)
df.to_csv("database/targets_SC.csv", columns=['id', 'tc'], index=False)