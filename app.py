import base64
import io

import fitz
from PIL import Image

import streamlit as st
from streamlit_extras.switch_page_button import switch_page

from FileReader.pdfFile import PDFDBStore


@st.cache_resource
def save_pdf_image(uploaded_file):
    pdf_bytes = uploaded_file.getvalue()
    st.session_state.pdf_bytes = pdf_bytes
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for i, page in enumerate(doc):  # iterate through the pages
        zoom_x = 2  # horizontal zoom
        zoom_y = 2  # vertical zoom
        mat = fitz.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension
        pix = page.get_pixmap(matrix=mat)  # use 'mat' instead of the identity matrix
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    # Combine images vertically
    long_image = Image.new("RGB", (images[0].width, sum(img.height for img in images)))
    y_offset = 0
    for img in images:
        long_image.paste(img, (0, y_offset))
        y_offset += img.height

    buffered = io.BytesIO()
    long_image.save(buffered, format="PNG")
    image_bytes = base64.b64encode(buffered.getvalue()).decode()

    return image_bytes


@st.cache_resource
def save_vector_store(_db):
    return _db.get_vectorDB(return_docs=True)


def initialize_session_state():
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = None

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "document_chunks" not in st.session_state:
        st.session_state.document_chunks = None

    if "pdf_image" not in st.session_state:
        st.session_state.pdf_image = None

    if "pdf_bytes" not in st.session_state:
        st.session_state.pdf_bytes = None

def set_openai_api_key(api_key):
    st.session_state["openai_api_key"] = api_key


def sidebar():
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑\n"  # noqa: E501
            "2. Upload a PDF file📄\n"
        )
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Paste your OpenAI API key here (sk-...)",
            help="You can get your API key from https://platform.openai.com/account/api-keys.",  # noqa: E501
            value=st.session_state.get("OPENAI_API_KEY", ""),
        )

        if api_key_input:
            set_openai_api_key(api_key_input)

        st.markdown("---")
        st.markdown("# About")
        st.markdown(
            "This tool allows you to chat with your "
            "documents as well as directly get Google and Youtube search results. "
        )
        st.markdown(
            "This tool is a work in progress. "
            "You can contribute to the project on [GitHub]() "
            "with your feedback and suggestions💡"
        )
        st.markdown("Made by [Arjun]()")
        st.markdown("---")


st.set_page_config(page_title="Research Paper Reading Assist Tool",
                   layout="centered",
                   initial_sidebar_state="expanded")

st.markdown("<style> ul {display: none;} </style>", unsafe_allow_html=True)  # Removes Page Navigation

st.title("DocumentGPT 📄")
sidebar()
initialize_session_state()

# Page 1 - Upload PDF
st.header("Upload your PDF document")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:

    pdfDB = PDFDBStore(uploaded_file)

    if st.session_state.openai_api_key is None:
        st.error("Please enter your OpenAI API key in the sidebar to continue.")

    elif not pdfDB.is_valid_key():
        st.error("Invalid OpenAI API key. Please enter a valid key in the sidebar to continue.")

    else:
        st.success("OpenAI API key set successfully!")

        with st.spinner("Processing PDF File...This may take a while⏳"):
            st.session_state.vector_store, st.session_state.document_chunks = save_vector_store(pdfDB)
            st.session_state.pdf_image = save_pdf_image(uploaded_file)

        st.success("PDF uploaded successfully!")
        st.session_state.vector_store, st.session_state.document_chunks = save_vector_store(pdfDB)
        st.session_state.pdf_image = save_pdf_image(uploaded_file)
        switch_page("results")
