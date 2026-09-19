# Assistant IA - Cours (RAG)

Assistant conversationnel qui répond aux questions des étudiants en se basant uniquement sur le contenu réel de leurs supports de cours (PDF), via une architecture **RAG (Retrieval-Augmented Generation)**. Projet réalisé dans le cadre du PFA 4ème année Big Data & IA - UIR.

## Fonctionnalités

- Authentification (connexion / création de compte)
- Upload de documents PDF depuis l'interface
- Indexation automatique du contenu (extraction, découpage, embeddings)
- Chat avec réponses basées uniquement sur les documents fournis
- Historique des conversations par utilisateur
- Choix du modèle IA (léger / complet)

## Architecture

1. **Extraction** - les PDF sont découpés en chunks de texte (`ingestion.py`)
2. **Indexation** - les chunks sont transformés en vecteurs (embeddings) et stockés dans une base FAISS (`embeddings.py`)
3. **Recherche** - à chaque question, les chunks les plus pertinents sont retrouvés par similarité vectorielle
4. **Génération** - un LLM local (via Ollama) génère la réponse à partir du contexte retrouvé (`rag_pipeline.py`)

## Stack technique

- **Frontend** : Streamlit
- **Embeddings** : sentence-transformers (`all-MiniLM-L6-v2`)
- **Base vectorielle** : FAISS
- **LLM** : Ollama (llama3 / llama3.2), 100% local, sans API payante
- **Authentification** : streamlit-authenticator + bcrypt

## Installation

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Installer [Ollama](https://ollama.com) puis télécharger un modèle :
```bash
ollama pull llama3.2
```

## Utilisation

```bash
streamlit run app.py
```

1. Créer un compte ou se connecter
2. Ajouter des documents PDF via Paramètres
3. Poser des questions sur les cours indexés

## Limites connues

- Le système peut occasionnellement générer des réponses approximatives sur des sujets proches du corpus mais non couverts par les documents indexés
- Performances dépendantes du matériel local (pas de GPU dédié dans cette configuration)
- Corpus de test limité (cours IoT et NLP)

## Auteur

Mohamed Amine JELOUL - UIR, 4ème année Big Data & IA