import streamlit as st
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities import Validator
import bcrypt
import yaml
from yaml.loader import SafeLoader
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from ingestion import charger_tous_les_documents
from embeddings import creer_index
from rag_pipeline import charger_index_et_chunks, poser_question
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Assistant Cours", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .stApp {
        background-color: #0e1117;
    }

    h1, h2, h3 {
        color: #f0f0f0 !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #2a2f3a;
    }

    /* Champs de saisie : fond sombre mais texte bien lisible */
    .stTextInput input,
    .stTextArea textarea,
    [data-testid="stChatInput"] textarea {
        background-color: #1c2129 !important;
        color: #f0f0f0 !important;
        border: 1px solid #3a4150 !important;
        border-radius: 8px !important;
    }
    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #8a8f98 !important;
    }
    .stTextInput label,
    .stTextArea label {
        color: #d0d0d0 !important;
    }

    .stButton > button {
        border-radius: 8px;
        border: 1px solid #3a4150;
        background-color: #1c2129;
        color: #f0f0f0;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #d97757;
        color: #d97757;
    }

    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.5rem;
        background-color: #1c2129;
        border: 1px solid #2a2f3a;
    }

    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        color: #8a8f98;
    }
    .stTabs [aria-selected="true"] {
        color: #d97757 !important;
        border-bottom-color: #d97757 !important;
    }

    @keyframes flotter {
        0%, 100% { transform: translateY(0px) rotate(-3deg); }
        50% { transform: translateY(-10px) rotate(3deg); }
    }
    .icone-flottante {
        display: inline-block;
        animation: flotter 3s ease-in-out infinite;
        font-size: 3rem;
        margin: 0 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)


class ValidateurSouple(Validator):
    def validate_username(self, username: str) -> bool:
        return True

    def validate_password(self, password: str) -> bool:
        return len(password) >= 4


authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    validator=ValidateurSouple(),
)


@st.cache_resource
def charger_modele_embeddings():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def charger_ressources_rag():
    if not os.path.exists("data/processed/index.faiss"):
        return None, None
    index, chunks = charger_index_et_chunks()
    return index, chunks


# ============================================================
# UTILISATEUR NON CONNECTE
# ============================================================
if not st.session_state.get("authentication_status"):

    if "mode_auth" not in st.session_state:
        st.session_state.mode_auth = None

    st.markdown(
        "<div style='text-align:center; margin-top:2rem;'>"
        "<span class='icone-flottante'>📚</span>"
        "<span class='icone-flottante' style='animation-delay:0.5s;'>✏️</span>"
        "<span class='icone-flottante' style='animation-delay:1s;'>📖</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h2 style='text-align:center; margin-top:0.5rem;'>Assistant IA - Cours</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:#8a8f98;'>Pose tes questions, comprends tes cours</p>",
        unsafe_allow_html=True,
    )

    col_gauche, col_centre, col_droite = st.columns([1, 2, 1])

    with col_centre:

        if st.session_state.mode_auth is None:
            st.write("")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Se connecter", use_container_width=True):
                    st.session_state.mode_auth = "connexion"
                    st.rerun()
            with col_b:
                if st.button("Creer un compte", use_container_width=True):
                    st.session_state.mode_auth = "inscription"
                    st.rerun()

        elif st.session_state.mode_auth == "connexion":
            if st.button("← Retour"):
                st.session_state.mode_auth = None
                st.rerun()

            with st.form("formulaire_connexion"):
                saisie_username = st.text_input("Username")
                saisie_email = st.text_input("Email")
                saisie_password = st.text_input("Mot de passe", type="password")
                bouton_connexion = st.form_submit_button("Connexion", use_container_width=True)

            if bouton_connexion:
                utilisateurs = config['credentials']['usernames']
                cle_trouvee = None
                for cle in utilisateurs:
                    if cle.lower() == saisie_username.lower():
                        cle_trouvee = cle
                        break

                if cle_trouvee is None:
                    st.error("Nom d'utilisateur ou email ou mot de passe incorrect")
                else:
                    donnees_utilisateur = utilisateurs[cle_trouvee]
                    email_correct = (donnees_utilisateur['email'] == saisie_email)
                    mdp_correct = bcrypt.checkpw(
                        saisie_password.encode('utf-8'),
                        donnees_utilisateur['password'].encode('utf-8')
                    )

                    if email_correct and mdp_correct:
                        nom_affiche = donnees_utilisateur.get('name')
                        if not nom_affiche:
                            nom_affiche = f"{donnees_utilisateur.get('first_name', '')} {donnees_utilisateur.get('last_name', '')}".strip()

                        st.session_state["authentication_status"] = True
                        st.session_state["name"] = nom_affiche
                        st.session_state["username"] = cle_trouvee
                        st.session_state["email"] = donnees_utilisateur['email']
                        st.rerun()
                    else:
                        st.error("Nom d'utilisateur ou email ou mot de passe incorrect")

        elif st.session_state.mode_auth == "inscription":
            if st.button("← Retour"):
                st.session_state.mode_auth = None
                st.rerun()

            try:
                email, username, name = authenticator.register_user(captcha=False)
                if email:
                    with open('config.yaml', 'w') as file:
                        yaml.dump(config, file, default_flow_style=False)
                    st.success("Compte cree avec succes ! Redirection...")
                    st.session_state.mode_auth = None
                    st.rerun()
            except Exception as e:
                st.error(f"Erreur : {e}")

# ============================================================
# UTILISATEUR CONNECTE
# ============================================================
else:
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if "conversation_active" not in st.session_state:
        st.session_state.conversation_active = None

    with st.sidebar:
        st.markdown(f"### 📖 {st.session_state['name']}")
        st.caption(st.session_state.get("email", ""))
        st.divider()

        if st.button("✏️ Nouvelle conversation", use_container_width=True):
            st.session_state.conversation_active = None
            st.rerun()

        st.markdown("**📚 Historique**")
        if not st.session_state.conversations:
            st.caption("Aucune conversation pour le moment")
        else:
            for titre in list(st.session_state.conversations.keys()):
                if st.button(titre, key=f"conv_{titre}", use_container_width=True):
                    st.session_state.conversation_active = titre
                    st.rerun()

        st.divider()

        with st.expander("⚙️ Parametres"):
            st.markdown("**Modele IA**")
            modele_choisi = st.selectbox(
                "Choisir le modele",
                options=["llama3.2", "llama3"],
                index=0,
                help="llama3.2 : plus rapide. llama3 : plus complet mais plus lent.",
            )
            st.session_state["modele_llm"] = modele_choisi

            st.divider()
            st.markdown("**📎 Ajouter des documents**")
            fichiers_pdf = st.file_uploader(
                "Ajouter des cours (PDF)", type="pdf", accept_multiple_files=True, key="upload_parametres"
            )
            if fichiers_pdf and st.button("Indexer les documents", key="indexer_parametres"):
                os.makedirs("data/raw", exist_ok=True)
                for f in fichiers_pdf:
                    with open(os.path.join("data/raw", f.name), "wb") as out:
                        out.write(f.getbuffer())
                with st.spinner("Indexation en cours..."):
                    modele_emb = charger_modele_embeddings()
                    creer_index(modele_emb)
                st.cache_resource.clear()
                with st.spinner("Preparation du chat..."):
                    charger_ressources_rag()
                st.success("Documents indexes !")

            st.divider()
            if st.button("Deconnexion", use_container_width=True):
                for cle in ["authentication_status", "name", "username", "email"]:
                    st.session_state.pop(cle, None)
                st.session_state.mode_auth = None
                st.rerun()

    index, chunks = charger_ressources_rag()

    titre_actif = st.session_state.conversation_active

    if index is None:
        st.markdown(
            "<div style='text-align:center; margin-top:3rem;'>"
            "<span class='icone-flottante'>📚</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<h2 style='text-align:center;'>Bonjour, {st.session_state['name']}</h2>",
            unsafe_allow_html=True,
        )
        st.info("Aucun document indexe pour le moment. Va dans Parametres pour ajouter tes cours PDF.")
    else:
        if titre_actif is None:
            st.markdown(
                "<div style='text-align:center; margin-top:3rem;'>"
                "<span class='icone-flottante'>📖</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<h2 style='text-align:center;'>Bonjour, {st.session_state['name']}</h2>",
                unsafe_allow_html=True,
            )
            messages = []
        else:
            messages = st.session_state.conversations[titre_actif]
            for msg in messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

    question = st.chat_input("Pose une question sur tes cours...")

    if question:
        if index is None:
            st.warning("Ajoute d'abord des documents via Parametres avant de poser une question.")
        else:
            if titre_actif is None:
                titre_actif = question[:40]
                st.session_state.conversations[titre_actif] = []
                st.session_state.conversation_active = titre_actif

            messages = st.session_state.conversations[titre_actif]
            messages.append({"role": "user", "content": question})

            with st.spinner("Reflexion en cours..."):
                modele_embeddings = charger_modele_embeddings()
                modele_a_utiliser = st.session_state.get("modele_llm", "llama3.2")
                reponse, sources = poser_question(question, index, chunks, modele_embeddings, modele_a_utiliser)

            messages.append({"role": "assistant", "content": reponse})
            st.rerun()