import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

stauth.Hasher.hash_passwords(config['credentials'])

with open('config.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)

print("Mots de passe haches avec succes.")