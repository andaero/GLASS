import pandas as pd
import numpy as np
import os
import glob
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

path_predictions_csv = './predictions/03_13_01-55/test_predictions_mp_onehot_MP_8_12_100_.csv'
path_actual_csv = './database/targets_MEGNet_2019.csv'

df_predictions = pd.read_csv(path_predictions_csv)
#change column name from 'name' to 'id'
df_predictions = df_predictions.rename(columns={"name": "id"})
df_actual = pd.read_csv(path_actual_csv)
df_combined = pd.merge(df_actual, df_predictions, on='id', how='inner')
#subset df_combined that only contains the last 10% of the rows
df_combined = df_combined.tail(int(len(df_combined) * 0.1))
print(df_combined)
#export dataframe to csv
df_combined.to_csv('FormationEnergyPredictedVsActual.csv', index=False)