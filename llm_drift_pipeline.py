import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

model = SentenceTransformer('all-MiniLM-L6-v2')

def detect_llm_drift(ref_path: str, cur_path: str, distance_threshold: float = 0.15):
    ref_df = pd.read_csv(ref_path)
    cur_df = pd.read_csv(cur_path)
    
    ref_texts = ref_df["text"].tolist()
    cur_texts = cur_df["text"].tolist()

    print("Encoding reference texts...")
    ref_embeddings = model.encode(ref_texts)
    
    print("Encoding current texts...")
    cur_embeddings = model.encode(cur_texts)

    ref_centroid = np.mean(ref_embeddings, axis=0).reshape(1, -1)
    cur_centroid = np.mean(cur_embeddings, axis=0).reshape(1, -1)

    distance = cosine_distances(ref_centroid, cur_centroid)[0][0]

    report = {
        "dataset_drift": bool(distance > distance_threshold),
        "metrics": {
            "cosine_distance": float(distance),
            "threshold": float(distance_threshold)
        }
    }
    
    return report

if __name__ == "__main__":
    drift_report = detect_llm_drift("data/llm_reference.csv", "data/llm_current.csv")
    print("\n LLM Drift Report ")
    print(json.dumps(drift_report, indent=4))