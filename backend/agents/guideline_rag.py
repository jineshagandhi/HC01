import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
with open("tfidf_vectorizer.pkl", "rb") as vf:
    vectorizer = pickle.load(vf)
with open("tfidf_matrix.pkl", "rb") as mf:
    tfidf_matrix = pickle.load(mf)
with open("docs.pkl", "rb") as df:
    docs = pickle.load(df)
with open("sources.pkl", "rb") as sf:
    sources = pickle.load(sf)
def get_guide(query, top=2):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(sims)[-top:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            'text': docs[idx],
            'source': sources[idx],
            'relevance': round(float(sims[idx]), 2)})
    return results
if __name__ == "__main__":
    q = "Patient has fever, WBC 22, lactate 4.5"
    res = get_guide(q)
    for r in res:
        print(f"[{r['source']} | rel={r['relevance']}]\n{r['text']}\n")