import os
from pypdf import PdfReader

RAW_DIR = "data/raw"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150   # chevauchement entre chunks pour ne pas couper une idee


def extraire_texte_pdf(chemin_pdf):
    """Lit un PDF et retourne tout son texte."""
    reader = PdfReader(chemin_pdf)
    texte = ""
    for page in reader.pages:
        texte += page.extract_text() + "\n"
    return texte


def decouper_en_chunks(texte, taille=CHUNK_SIZE, chevauchement=CHUNK_OVERLAP):
    """Decoupe un long texte en morceaux avec un leger chevauchement."""
    chunks = []
    debut = 0
    while debut < len(texte):
        fin = debut + taille
        chunks.append(texte[debut:fin])
        debut += taille - chevauchement
    return chunks


def charger_tous_les_documents():
    """Parcourt data/raw, extrait et decoupe tous les PDF trouves."""
    tous_les_chunks = []

    if not os.path.exists(RAW_DIR):
        print(f"Le dossier {RAW_DIR} n'existe pas.")
        return tous_les_chunks

    fichiers = [f for f in os.listdir(RAW_DIR) if f.endswith(".pdf")]

    if not fichiers:
        print(f"Aucun PDF trouve dans {RAW_DIR}. Ajoute tes cours et relance.")
        return tous_les_chunks

    for nom_fichier in fichiers:
        chemin = os.path.join(RAW_DIR, nom_fichier)
        print(f"Lecture de {nom_fichier}...")
        texte = extraire_texte_pdf(chemin)
        chunks = decouper_en_chunks(texte)
        print(f"  -> {len(chunks)} chunks extraits")
        tous_les_chunks.extend(chunks)

    return tous_les_chunks


if __name__ == "__main__":
    chunks = charger_tous_les_documents()
    print(f"\nTotal : {len(chunks)} chunks extraits de tous les documents.")
    if chunks:
        print("\nExemple du premier chunk :")
        print(chunks[0])