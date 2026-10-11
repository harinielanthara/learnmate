import streamlit as st

import views
from api import call

st.set_page_config(page_title="LearnMate", page_icon="🎓", layout="wide")
views.inject_css()

if "token" not in st.session_state:
    views.login_page()
    st.stop()

r = call("GET", "/auth/me")
st.session_state["me"] = r.json() if r.ok else {}

home = st.Page(views.home, title="Home", icon="🏠", url_path="home", default=True)
chat = st.Page(views.chat, title="Chat", icon="💬", url_path="chat")
library = st.Page(views.library, title="Library", icon="📚", url_path="library")
concepts = st.Page(views.concepts, title="Concept map", icon="🧩", url_path="concepts")
views.PAGES.update(home=home, chat=chat, library=library, concepts=concepts)

nav = st.navigation([home, chat, library, concepts])
views.sidebar()
nav.run()
