import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
docs = []
sources = []
for fname in os.listdir("guidelines"):
    if fname.endswith(".txt"):
        with open(os.path.join("guidelines", fname), 'r', encoding='utf-8') as f:
            txt = f.read()
        chunks = [c.strip() for c in txt.split('\n\n') if c.strip()]
        for chunk in chunks:
            docs.append(chunk)
            sources.append(fname)
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(docs)
with open("tfidf_vectorizer.pkl", "wb") as vf:
    pickle.dump(vectorizer, vf)
with open("tfidf_matrix.pkl", "wb") as mf:
    pickle.dump(tfidf_matrix, mf)
with open("docs.pkl", "wb") as df:
    pickle.dump(docs, df)
with open("sources.pkl", "wb") as sf:
    pickle.dump(sources, sf)
print(f"Indexed {len(docs)} chunks from guidelines/")