from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import ollama

MODELE_EMBEDDINGS = "all-MiniLM-L6-v2"
INDEX_PATH = "data/processed/index.faiss"
CHUNKS_PATH = "data/processed/chunks.pkl"
MODELE_LLM = "llama3.2"
NB_CHUNKS_A_RECUPERER = 5


def charger_index_et_chunks():
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def rechercher_chunks_pertinents(question, index, chunks, modele_embeddings, k=NB_CHUNKS_A_RECUPERER):
    vecteur_question = modele_embeddings.encode([question]).astype("float32")
    distances, indices = index.search(vecteur_question, k)
    chunks_trouves = [chunks[i] for i in indices[0]]
    return chunks_trouves


def construire_prompt(question, chunks_contexte):
    contexte = "\n\n---\n\n".join(chunks_contexte)
    prompt = f"""Tu es un assistant pedagogique qui explique clairement les concepts de cours a un etudiant, en te basant UNIQUEMENT sur le contexte fourni ci-dessous.
Donne une explication complete et structuree, avec des exemples si le contexte en fournit.

REGLE STRICTE : si le contexte ne contient pas d'information pertinente pour repondre, dis simplement "Cette information ne figure pas dans les documents fournis." et ARRETE-TOI LA. N'ajoute jamais de connaissances generales en dehors du contexte, meme partiellement.

Contexte :
{contexte}

Question : {question}

Explication :"""
    return prompt


def poser_question(question, index, chunks, modele_embeddings, modele_llm=MODELE_LLM):
    chunks_pertinents = rechercher_chunks_pertinents(question, index, chunks, modele_embeddings)
    prompt = construire_prompt(question, chunks_pertinents)

    reponse = ollama.chat(
        model=modele_llm,
        messages=[{"role": "user", "content": prompt}],
        keep_alive="30m"
    )

    return reponse["message"]["content"], chunks_pertinents


if __name__ == "__main__":
    print("Chargement de l'index et des chunks...")
    index, chunks = charger_index_et_chunks()

    print(f"Chargement du modele d'embeddings ({MODELE_EMBEDDINGS})...")
    modele_embeddings = SentenceTransformer(MODELE_EMBEDDINGS)

    print("\nPret ! Tape ta question (ou 'quit' pour sortir)\n")

    while True:
        question = input("Question : ")
        if question.lower() in ["quit", "exit", "q"]:
            break

        reponse, sources = poser_question(question, index, chunks, modele_embeddings)

        print(f"\nReponse : {reponse}\n")
        print("Sources utilisees :")
        for i, chunk in enumerate(sources, 1):
            print(f"  [{i}] {chunk[:100]}...")
        print()