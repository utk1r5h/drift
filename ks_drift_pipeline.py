import pandas as pd
import numpy as np
from alibi_detect.cd import KSDrift
import warnings

warnings.filterwarnings('ignore', module='alibi_detect')

def detect_tabular_shift(ref_path: str, curr_path: str, p_val_threshold: float = 0.05):
  ref_df = pd.read_csv(ref_path)
  curr_df = pd.read_csv(curr_path)

  feature_names = ref_df.columns.to_list()

  cd = KSDrift(
    x_ref = ref_df.values,
    p_val = p_val_threshold
  )

  preds = cd.predict(
    curr_df.values,
    return_p_val = True,
    return_distance = True
  )

  report ={
    "dataset_drift": bool(preds["data"]["is_drift"]==1),
    "features": {}
  }

  for i, feature in enumerate(feature_names):
    report["features"][feature]={
      "drift_detected": bool(preds["data"]["p_val"][i]<preds["data"]["threshold"]),
      "p_value": float(preds["data"]["p_val"][i]),
      "ks_stats": float(preds["data"]["distance"][i])
    }
  return report

if __name__ == "__main__":
    drift_report = detect_tabular_shift("data/reference.csv", "data/current.csv")
    
    # Pretty print the dictionary so it looks like a JSON API response
    import json
    print(json.dumps(drift_report, indent=4))