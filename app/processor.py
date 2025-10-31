import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pennylane as qml

# Model file paths
VECTORIZER_FILE = "models/vectorizer.pkl"
TFIDF_MATRIX_FILE = "models/tfidf_matrix.pkl"
POLICIES_FILE = "models/policies.pkl"

os.makedirs("models", exist_ok=True)

def load_or_train_models(csv_path: str = "policies.csv"):
    """Load trained models if available, else train from CSV."""
    if all(os.path.exists(f) for f in [VECTORIZER_FILE, TFIDF_MATRIX_FILE, POLICIES_FILE]):
        vectorizer = joblib.load(VECTORIZER_FILE)
        tfidf_matrix = joblib.load(TFIDF_MATRIX_FILE)
        policies = joblib.load(POLICIES_FILE)
        print("✅ Loaded existing models from disk.")
        return vectorizer, tfidf_matrix, policies

    print("⚙️ Training new models from policies.csv...")
    policies = pd.read_csv(csv_path)
    text_data = policies["full_text"].astype(str).tolist()

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(text_data)

    joblib.dump(vectorizer, VECTORIZER_FILE)
    joblib.dump(tfidf_matrix, TFIDF_MATRIX_FILE)
    joblib.dump(policies, POLICIES_FILE)
    print("✅ Models trained and saved.")

    return vectorizer, tfidf_matrix, policies


def quantum_similarity(vec1, vec2):
    """Compute similarity using a small PennyLane quantum circuit."""
    n_qubits = min(4, len(vec1))
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(v1, v2):
        for i in range(n_qubits):
            qml.RY(v1[i % len(v1)] * np.pi, wires=i)
        qml.Barrier()
        for i in range(n_qubits):
            qml.RY(-v2[i % len(v2)] * np.pi, wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    result = circuit(vec1, vec2)
    return float(np.mean(result))


def answer_query(query: str, top_k: int = 5):
    """Return top-k most similar policy results."""
    vectorizer, tfidf_matrix, policies = load_or_train_models()

    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = similarities.argsort()[-top_k:][::-1]
    results = []

    for i in top_indices:
        row = policies.iloc[i]
        results.append({
            "policy_id": row["policy_id"],
            "title": row["title"],
            "region": row["region"],
            "year": row["year"],
            "status": row["status"],
            "summary": row["full_text"][:200] + "...",
            "similarity": round(float(similarities[i]), 3)
        })

    return results
