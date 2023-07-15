import streamlit as st
import streamlit.components.v1 as components
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header

from youtube_search import YoutubeSearch

import utils
import base64

st.set_page_config(page_title="Display Results",
                   layout="wide",
                   initial_sidebar_state="collapsed")

utils.load_tab_css()
utils.initialize_session_state()

######################################################################

st.title("Have fun Researching! 📚")

tab1, gap1, tab2, gap2, tab3, gap3, tab4 = st.tabs(["PDF", "   ", "Google", "   ", "Youtube", " ", "Chat"])

with tab1:
    st.subheader("PDF Display")
    col1, col2, col3 = st.columns([0.5, 9, 0.5])

    with col2:
        container_height = 700
        image_bytes = st.session_state.pdf_image
        img_html = f'<img src="data:image/png;base64,{image_bytes}"/>'
        # st.markdown(
        #     f'<div style="height: {container_height}px; overflow-y: scroll; overflow-x: hidden; text-align: center; '
        #     f'border: 5px solid #888888; border-radius: 4px;">{img_html}</div>',
        #     unsafe_allow_html=True,
        # )

        components.html(
            f'<div style="height: {container_height}px; overflow-y: scroll; overflow-x: hidden; text-align: center; '
            f'border: 5px solid #888888; border-radius: 4px;">{img_html}</div>'
          )

        # pdf_bytes = st.session_state["pdf_bytes"]
        # base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        # pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="900" height="800" ' \
        #               f'type="application/pdf"></iframe> '
        # st.markdown(pdf_display, unsafe_allow_html=True)

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
    results = []
    for i, keyword in enumerate(st.session_state.search_keywords):
        buttons.append(cols[i].button(keyword, use_container_width=True, key=i))

    search_y = st.text_input("What do you want to search for?", key='youtube')

    for i, button in enumerate(buttons):
        if button:
            search_y = st.session_state.search_keywords[i]

    if search_y:
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
    st.text("Ask the chatbot anything! If the chatbot is unable to fetch the answer from the provided document,")
    st.text("it will also perform an additional Web Search to gather more context. You can view the sources below.")

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
            on_click=st.session_state.agent.run_callback,
        )

    buttons_placeholder = st.container()

    with buttons_placeholder:
        cols = st.columns([0.15, 1])

        cols[0].button("Regenerate Response",
                       key="regenerate",
                       on_click=st.session_state.agent.regenerate_response)

        cols[-1].button("Clear Chat",
                        key="clear",
                        on_click=st.session_state.agent.clear_conversation)

    with st.expander("View Web Sources"):

        colored_header(label="Web Searches", description="Related Web Search Results",
                       color_name="light-blue-70")

        if len(st.session_state.google_sources) != 0:

            for source in st.session_state.google_sources:
                if isinstance(source, dict):
                    source_text = f"Title: {source['title']}\n\nLink: {source['link']}\n\nSnippet: {source['snippet']}\n\nScraped Results: {source['answer']} "
                else:
                    source_text = source

                st.divider()
                st.write(source_text)

        else:

            st.write("No Web sources found")

    with st.expander("View Document Sources"):

        colored_header(label="Source Documents", description="Related Document Chunks", color_name="orange-70")

        if len(st.session_state.doc_sources) != 0:

            for document in st.session_state.doc_sources:
                st.divider()
                source_text = f"{document.page_content}\n\nPage Number: {document.metadata['page_number']}\nChunk: {document.metadata['chunk']}"
                st.write(source_text)

        else:

            st.write("No document sources found")

    utils.enterKeypress_submit_button_html()

    credit_card_placeholder.caption(f"""
    Used {st.session_state.token_count} tokens\n
    Cost {round((st.session_state.token_count / 1000) * 0.002, 6)} $ (approx.)
    """)
