import streamlit as st
from streamlit_extras.switch_page_button import switch_page

import re

import pypdf

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

from openai.error import OpenAIError, AuthenticationError


def initialize_session_state():
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = None

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "pdf_bytes" not in st.session_state:
        st.session_state.pdf_bytes = None


def set_openai_api_key(api_key):
    st.session_state["openai_api_key"] = api_key


def sidebar():
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Enter [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑\n"  # noqa: E501
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
        st.markdown("# What is DocumentGPT?")
        st.markdown(
            "This tool allows you to chat with your "
            "documents as well as directly get Google and Youtube search results. "
        )
        st.markdown(
            "Contribute to the project on [GitHub](https://github.com/aju22) 💡"
        )
        st.markdown("Made by [Arjun](https://twitter.com/ArjunKr81620614)")
        st.markdown("---")


class PDFDBStore:

    def __init__(self, pdf_file):

        self.reader = pypdf.PdfReader(pdf_file)
        self.embeddings = OpenAIEmbeddings(openai_api_key=st.session_state['openai_api_key'])
        self.vector_store = None

    def is_valid_key(self):

        try:
            embeddings = OpenAIEmbeddings(openai_api_key=st.session_state['openai_api_key'])
            embeddings.embed_query("test")

        except AuthenticationError as e:
            return False

        return True

    def extract_metadata_from_pdf(self):
        metadata = self.reader.metadata
        return {
            "title": metadata.get("/Title", "").strip(),
            "author": metadata.get("/Author", "").strip(),
            "creation_date": metadata.get("/CreationDate", "").strip(),
        }

    def extract_pages_from_pdf(self):
        """
        Extracts the text from each page of the PDF.
        :param file_path: The path to the PDF file.
        :return: A list of tuples containing the page number and the extracted text.
        """

        pages = []
        for page_num, page in enumerate(self.reader.pages):
            text = page.extract_text()
            if text.strip():  # Check if extracted text is not empty
                pages.append((page_num + 1, text))

        return pages

    def parse_pdf(self):
        """
        Extracts the title and text from each page of the PDF.
        :param file_path: The path to the PDF file.
        :return: A tuple containing the title and a list of tuples with page numbers and extracted text.
        """

        metadata = self.extract_metadata_from_pdf()
        pages = self.extract_pages_from_pdf()

        return pages, metadata

    def merge_hyphenated_words(self, text):
        return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    def fix_newlines(self, text):
        return re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    def remove_multiple_newlines(self, text):
        return re.sub(r"\n{2,}", "\n", text)

    def clean_text(self, pages, cleaning_functions):
        cleaned_pages = []
        for page_num, text in pages:
            for cleaning_function in cleaning_functions:
                text = cleaning_function(text)
            cleaned_pages.append((page_num, text))

        return cleaned_pages

    def text_to_docs(self, text, metadata):
        """Converts list of strings to a list of Documents with metadata."""
        doc_chunks = []

        for page_num, page in text:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                chunk_overlap=200,
            )
            chunks = text_splitter.split_text(page)
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "page_number": page_num,
                        "chunk": i,
                        "source": f"p{page_num}-{i}",
                        **metadata,
                    },
                )
                doc_chunks.append(doc)

        return doc_chunks

    def get_docChunks(self):

        raw_pages, metadata = self.parse_pdf()

        cleaning_functions = [
            self.merge_hyphenated_words,
            self.fix_newlines,
            self.remove_multiple_newlines,
        ]
        cleaned_text_pdf = self.clean_text(raw_pages, cleaning_functions)
        document_chunks = self.text_to_docs(cleaned_text_pdf, metadata)

        # Optional: Reduce embedding cost by only using the first 23 pages
        # document_chunks = document_chunks[:70]

        return document_chunks

    def get_vectorDB(self):

        document_chunks = self.get_docChunks()
        self.vector_store = FAISS.from_documents(
            document_chunks,
            self.embeddings
        )

        return self.vector_store


st.set_page_config(page_title="Research Paper Reading Assist Tool",
                   layout="centered",
                   initial_sidebar_state="expanded")

st.markdown("<style> ul {display: none;} </style>", unsafe_allow_html=True)

st.title("DocumentGPT 📄")
sidebar()
initialize_session_state()

# Page 1 - Upload PDF
st.header("Upload your PDF document")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")


if uploaded_file is not None:

    pdfDB = PDFDBStore(uploaded_file)
    st.session_state.pdf_bytes = uploaded_file.getvalue()

    if st.session_state.openai_api_key is None:
        st.error("Please enter your OpenAI API key in the sidebar to continue.")

    elif not pdfDB.is_valid_key():
        st.error("Invalid OpenAI API key. Please enter a valid key in the sidebar to continue.")

    else:
        st.success("OpenAI API key set successfully!")

        with st.spinner("Processing PDF File...This may take a while⏳"):
            st.session_state.vector_store = pdfDB.get_vectorDB()

        st.success("PDF uploaded successfully!")
        st.session_state.pdf_bytes = uploaded_file.getvalue()
        switch_page("results")
