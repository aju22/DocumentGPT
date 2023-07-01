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
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.llm import LLMChain


@dataclass
class Message:
    """Class for keeping track of a chat message."""
    role: Literal["human", "ai"]
    content: str


class State:
    def __init__(self):
        pass

    def initialize_all(self):
        self.initialize_session_state()
        self.load_tab_css()

    def load_tab_css(self):
        with open("static/tab.css", "r") as f:
            css = f"<style>{f.read()}</style>"
            st.markdown(css, unsafe_allow_html=True)

    def initialize_session_state(self):
        if "history" not in st.session_state:
            st.session_state.history = []
        if "memory" not in st.session_state:
            st.session_state.memory = []
        if "search_keywords" not in st.session_state:
            st.session_state.search_keywords = []
        if "token_count" not in st.session_state:
            st.session_state.token_count = 0
        if "chat_placeholder" not in st.session_state:
            st.session_state.chat_placeholder = None
        if "chat_container" not in st.session_state:
            st.session_state.chat_output = None
        if "chat_text" not in st.session_state:
            st.session_state.text = None


class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token

        with st.session_state.chat_container:
            st.session_state.chat_text.write(self.text)

    def reset(self):
        self.text = ""


class ConversationalBot:

    def __init__(self, stream_handler: StreamHandler):
        self.stream_handler = stream_handler
        self.conversation = self.get_model_chain()

    def get_model_chain(self):
        chat_model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature="0",
            openai_api_key=st.session_state['openai_api_key'],
            streaming=True,
            callbacks=[self.stream_handler],
            verbose=False
        )

        summary_model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature="0",
            openai_api_key=st.session_state['openai_api_key'],
            verbose=False
        )

        question_generator = LLMChain(llm=summary_model, prompt=CONDENSE_QUESTION_PROMPT)
        doc_chain = load_qa_chain(chat_model, chain_type="stuff", prompt=QA_PROMPT)

        return ConversationalRetrievalChain(
            retriever=st.session_state.vector_store.as_retriever(),
            combine_docs_chain=doc_chain,
            question_generator=question_generator,
            return_source_documents=False,
            # verbose=True
        )

    def run_chain(self):

        self.stream_handler.reset()

        with get_openai_callback() as cb:
            st.session_state.token_count += cb.total_tokens
            print(f"Total tokens: {st.session_state.token_count}")

            with st.session_state.chat_placeholder:
                human = st.chat_message("user")
                human.write(st.session_state.human_prompt)

            with st.session_state.chat_placeholder:
                st.session_state.chat_container = st.chat_message('assistant')
                with st.session_state.chat_container:
                    st.session_state.chat_text = st.empty()

            return self.conversation(
                {"question": st.session_state.human_prompt, "chat_history": st.session_state.memory}
            )['answer']

    def store_conversation(self, llm_response):

        st.session_state.memory.append((st.session_state.human_prompt, llm_response))
        st.session_state.search_keywords += self.get_keywords()

        st.session_state.history.append(
            Message("human", st.session_state.human_prompt)
        )
        st.session_state.history.append(
            Message("ai", llm_response)
        )

        st.session_state.human_prompt = ""

        st.session_state.memory = st.session_state.memory[-3:]
        st.session_state.search_keywords = st.session_state.search_keywords[-5:]

        print(f"Search keywords: {st.session_state.search_keywords}")

    def get_keywords(self):

        conversation = ""
        keyword_list = []

        for human_prompt, llm_response in st.session_state.memory:
            conversation += human_prompt + "\n"
            conversation += llm_response + "\n"

        openai.api_key = st.session_state['openai_api_key']

        search_keywords_extract_function = {
            "name": "search_keywords_extractor",
            "description": "Creates a list of 5 short academic Google searchable keywords from the given conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "string",
                        "description": "List of 5 short academic Google searchable keywords"
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

        if "function_call" in res['choices'][0]['message']:
            args = json.loads(res['choices'][0]['message']['function_call']['arguments'])
            keyword_list = list(args['keywords'].split(","))

        return keyword_list

    def run_callback(self):

        llm_response = self.run_chain()
        self.store_conversation(llm_response)


def run_html():
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


st.set_page_config(page_title="Display Results",
                   layout="wide",
                   initial_sidebar_state="collapsed")

State().initialize_all()
bot = ConversationalBot(StreamHandler())

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
    st.subheader("Chatbot")

    st.session_state.chat_placeholder = st.container()
    prompt_placeholder = st.form("chat-form")
    credit_card_placeholder = st.empty()

    with st.session_state.chat_placeholder:
        for chat in st.session_state.history:
            if chat.role == "human":
                user = st.chat_message("user")
                user.write(chat.content)
            else:
                ai = st.chat_message("assistant")
                ai.write(chat.content)

    with prompt_placeholder:
        st.markdown("**Chat**")
        cols = st.columns((6, 1))
        text_input = cols[0].text_input(
            "Chat",
            value="",
            label_visibility="collapsed",
            key="human_prompt",
        )
        cols[1].form_submit_button(
            "Submit",
            type="primary",
            on_click=bot.run_callback,
        )

    run_html()

    credit_card_placeholder.caption(f"""
    Used {st.session_state.token_count} tokens
    """)
