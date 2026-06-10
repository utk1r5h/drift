import pandas as pd
import numpy as np
import warnings
from alibi_detect.cd import KSDrift
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

warnings.filterwarnings('ignore', module='alibi_detect')



class UnifiedDriftEngine:
    def __init__(self):

        self._llm_model = None

    @property
    def llm_model(self):
        if self._llm_model is None:
            print("Loading SentenceTransformer ('all-MiniLM-L6-v2')...")
            self._llm_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._llm_model

    def analyze_tabular(self, ref_data: list[dict], cur_data: list[dict], p_val_threshold: float = 0.05) -> dict:

        ref_df = pd.DataFrame(ref_data)
        cur_df = pd.DataFrame(cur_data)
        feature_names = ref_df.columns.tolist()

        cd = KSDrift(x_ref=ref_df.values, p_val=p_val_threshold)
        preds = cd.predict(cur_df.values, return_p_val=True, return_distance=True)

        report = {
            "drift_type": "tabular",
            "dataset_drift": bool(preds["data"]["is_drift"] == 1),
            "global_threshold": float(preds["data"]["threshold"]),
            "features": {}
        }

        for i, feature in enumerate(feature_names):
            report["features"][feature] = {
                "drift_detected": bool(preds["data"]["p_val"][i] < preds["data"]["threshold"]),
                "p_value": float(preds["data"]["p_val"][i]),
                "statistic": float(preds["data"]["distance"][i])
            }
        return report

    def analyze_llm(self, ref_texts: list[str], cur_texts: list[str], distance_threshold: float = 0.15) -> dict:

        ref_embeddings = self.llm_model.encode(ref_texts)
        cur_embeddings = self.llm_model.encode(cur_texts)

        ref_centroid = np.mean(ref_embeddings, axis=0).reshape(1, -1)
        cur_centroid = np.mean(cur_embeddings, axis=0).reshape(1, -1)

        distance = cosine_distances(ref_centroid, cur_centroid)[0][0]

        return {
            "drift_type": "llm_semantic",
            "dataset_drift": bool(distance > distance_threshold),
            "metrics": {
                "cosine_distance": float(distance),
                "threshold": float(distance_threshold)
            }
        }

if __name__ == "__main__":
    import json
    engine = UnifiedDriftEngine()
    

    print("=" * 40)
    print(" RUNNING TABULAR DRIFT ANALYSIS")
    print("=" * 40)
    try:
        ref_tab = pd.read_csv("data/reference.csv").to_dict(orient="records")
        cur_tab = pd.read_csv("data/current.csv").to_dict(orient="records")
        tab_report = engine.analyze_tabular(ref_tab, cur_tab)
        print(json.dumps(tab_report, indent=4))
    except Exception as e:
        print(f"Tabular test failed: {e}")

    print("\n" + "=" * 40)
    print(" RUNNING LLM SEMANTIC DRIFT ANALYSIS")
    print("=" * 40)
    try:
        ref_txt = pd.read_csv("data/llm_reference.csv")["text"].tolist()
        cur_txt = pd.read_csv("data/llm_current.csv")["text"].tolist()
        llm_report = engine.analyze_llm(ref_txt, cur_txt)
        print(json.dumps(llm_report, indent=4))
    except Exception as e:
        print(f"LLM test failed: {e}")