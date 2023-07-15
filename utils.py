import streamlit as st
import streamlit.components.v1 as components
from Conversation.conversation import ConversationalAgent


def load_tab_css():
    with open("static/tab.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)


def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "search_keywords" not in st.session_state:
        st.session_state.search_keywords = []
    if "doc_sources" not in st.session_state:
        st.session_state.doc_sources = []
    if "google_sources" not in st.session_state:
        st.session_state.google_sources = []
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "chat_placeholder" not in st.session_state:
        st.session_state.chat_placeholder = None
    if "agent" not in st.session_state:
        st.session_state.agent = ConversationalAgent()


def enterKeypress_submit_button_html():
    return components.html("""
                            <script>
                                const streamlitDoc = window.parent.document;
                                const buttons = Array.from(
                                    streamlitDoc.querySelectorAll('.stButton > button')
                                );
                                const submitButton = buttons.find(
                                    el => el.innerText === 'Submit'
                                );
                                streamlitDoc.addEventListener('keydown', function(e) {
                                    switch (e.key) {
                                        case 'Enter':
                                            submitButton.click();
                                            break;
                                    }
                                });
                            </script>
                            """,
                           height=0,
                           width=0,
                           )
