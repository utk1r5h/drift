import pandas as pd
import numpy as np
from alibi_detect.cd import KSDrift

ref_df = pd.read_csv("data/reference.csv")
cur_df = pd.read_csv("data/current.csv")

x_ref = ref_df.values
x_curr = cur_df.values


cd = KSDrift(
  x_ref=x_ref,
  p_val = 0.05
)

preds = cd.predict(
  x_curr,
  return_p_val= True,
  return_distance = True 
)

print("Drift Detected:", preds["data"]["is_drift"])
print("P-values:", preds["data"]["p_val"])
print("KS Statistics:", preds["data"]["distance"])
print("Threshold:", preds["data"]["threshold"])



