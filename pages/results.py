import streamlit as st
import streamlit.components.v1 as components
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header

from youtube_search import YoutubeSearch

import base64
import json

from dataclasses import dataclass
from typing import Literal

import openai

from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain


@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["human", "ai"]
    message: str


def load_chat_css():
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)


def load_tab_css():
    with open("static/tab.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)


def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "memory" not in st.session_state:
        st.session_state.memory = []
    if "search_keywords" not in st.session_state:
        st.session_state.search_keywords = []
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "conversation" not in st.session_state:
        chat_model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature="0",
            openai_api_key=st.session_state['openai_api_key'],
            verbose=False
        )
        st.session_state.conversation = ConversationalRetrievalChain.from_llm(
            chat_model,
            retriever=st.session_state.vector_store.as_retriever(),
            condense_question_llm=chat_model,
            verbose=False
        )


def on_click_callback():
    with get_openai_callback() as cb:
        human_prompt = st.session_state.human_prompt
        llm_response = st.session_state.conversation(
            {"question": human_prompt, "chat_history": st.session_state.memory}
        )['answer']

        st.session_state.memory.append((human_prompt, llm_response))
        st.session_state.search_keywords += get_keywords()

        st.session_state.history.append(
            Message("human", human_prompt)
        )
        st.session_state.history.append(
            Message("ai", llm_response)
        )

        # st.session_state.human_prompt = ""

        st.session_state.memory = st.session_state.memory[-3:]
        st.session_state.search_keywords = st.session_state.search_keywords[-5:]
        st.session_state.token_count += cb.total_tokens


def get_keywords():
    conversation = ""
    for human_prompt, llm_response in st.session_state.memory:
        conversation += human_prompt + "\n"
        conversation += llm_response + "\n"

    openai.api_key = st.session_state['openai_api_key']

    search_keywords_extract_function = {
        "name": "search_keywords_extractor",
        "description": "Creates a list of 5 google/youtube academic searchable keywords from a given summary",
        "parameters": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": "List of 5 google/youtube academic searchable keywords"
                }
            },
            "required": ["keywords"]
        }
    }

    res = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-0613',
        messages=[{"role": "user", "content": conversation}],
        functions=[search_keywords_extract_function]
    )

    keyword_list = []

    if "function_call" in res['choices'][0]['message']:
        args = json.loads(res['choices'][0]['message']['function_call']['arguments'])
        keyword_list = list(args['keywords'].split(","))

    return keyword_list

st.set_page_config(page_title="Display Results",
                   layout="wide",
                   initial_sidebar_state="collapsed")
load_tab_css()
initialize_session_state()
st.title("Have fun Researching! 📚")

tab1, tabgap1, tab2, tagap2, tab3, tagap3, tab4 = st.tabs(["PDF", "   ", "Google", "   ", "Youtube", " ", "Chat"])

with tab1:
    st.subheader("PDF Display")
    col1, col2, col3 = st.columns([1.8, 7, 1])

    with col2:
        pdf_bytes = st.session_state["pdf_bytes"]
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="900" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

with tab2:
    st.subheader("Google Results")
    cols = st.columns(5)
    buttons = []
    for i, keyword in enumerate(st.session_state.search_keywords):
        buttons.append(cols[i].button(keyword, use_container_width=True))

    search_g = ""

    for i, button in enumerate(buttons):
        if button:
            search_g = st.session_state.search_keywords[i]

    components.iframe(f"https://www.google.com/search?igu=1&ei=&q={search_g}", height=1000, scrolling=True)

with tab3:
    st.subheader("Youtube Results")
    cols = st.columns(5)
    buttons = []
    for i, keyword in enumerate(st.session_state.search_keywords):
        buttons.append(cols[i].button(keyword, use_container_width=True, key=i))

    search_y = st.text_input("What do you want to search for?", key='youtube')

    for i, button in enumerate(buttons):
        if button:
            search_y = st.session_state.search_keywords[i]

    results = YoutubeSearch(search_y, max_results=3).to_dict()

    _, col2, _ = st.columns([1.8, 5, 1])

    with col2:
        for i in range(len(results)):
            colored_header(
                label=results[i]['title'],
                description=f"Duration: {results[i]['duration']} \t Views: {results[i]['views'][:-6]}  \t  Published: {results[i]['publish_time']}",
                color_name="blue-green-70",
            )
            st.video(f"https://www.youtube.com/watch?v={results[i]['id']}")

            add_vertical_space(5)

with tab4:
    load_chat_css()

    st.title("Chatbot 🤖")

    chat_placeholder = st.container()
    prompt_placeholder = st.form("chat-form")
    credit_card_placeholder = st.empty()

    with chat_placeholder:
        for chat in st.session_state.history:
            div = f"""
        <div class="chat-row 
        {'' if chat.origin == 'ai' else 'row-reverse'}">
            <img class="chat-icon" src="app/static/{
            'ai_icon.png' if chat.origin == 'ai'
            else 'user_icon.png'}"
             width=32 height=32>
            <div class="chat-bubble
        {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
            &#8203;{chat.message}
            </div>
        </div>
            """
            st.markdown(div, unsafe_allow_html=True)

        for _ in range(3):
            st.markdown("")

    with prompt_placeholder:
        st.markdown("**Chat**")
        cols = st.columns((6, 1))
        text_input = cols[0].text_input(
            "Chat",
            value="Hello Bot!",
            label_visibility="collapsed",
            key="human_prompt",
        )
        cols[1].form_submit_button(
            "Submit",
            type="primary",
            on_click=on_click_callback,
        )

    credit_card_placeholder.caption(f"""
    Used {st.session_state.token_count} tokens
    """)

components.html("""
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

