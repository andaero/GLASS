import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Prediction Evaluator")

parser.add_argument("--path", type=str,
    default='predictions/01_21-59/test_predictions_mp_onehot_MP_8_12_100_.csv')
parser.add_argument("--threshold", type=float, default=100)

options = parser.parse_args()
print(options.path)

df = pd.read_csv(options.path)

df_threshold = df[df['prediction'] > options.threshold]
print(df_threshold)