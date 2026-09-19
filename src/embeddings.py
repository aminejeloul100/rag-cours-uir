from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

from ingestion import charger_tous_les_documents

MODELE_EMBEDDINGS = "all-MiniLM-L6-v2"  # petit modele, rapide, gratuit, en local
INDEX_PATH = "data/processed/index.faiss"
CHUNKS_PATH = "data/processed/chunks.pkl"


def creer_index(modele=None):
    print("Chargement et decoupage des documents...")
    chunks = charger_tous_les_documents()

    if not chunks:
        print("Aucun chunk trouve, arret.")
        return

    if modele is None:
        print(f"Chargement du modele d'embeddings ({MODELE_EMBEDDINGS})...")
        modele = SentenceTransformer(MODELE_EMBEDDINGS)

    print(f"Generation des embeddings pour {len(chunks)} chunks...")
    embeddings = modele.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    print("Creation de l'index FAISS...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs("data/processed", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Index sauvegarde dans {INDEX_PATH}")
    print(f"Chunks sauvegardes dans {CHUNKS_PATH}")
    print(f"\nTermine ! {len(chunks)} chunks indexes.")


if __name__ == "__main__":
    creer_index()