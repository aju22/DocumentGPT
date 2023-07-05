import streamlit as st

import re

import pypdf

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

from openai.error import AuthenticationError


class PDFDBStore:
    """

    """

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
        :return: A tuple containing the title and a list of tuples with page numbers and extracted text.
        """

        metadata = self.extract_metadata_from_pdf()
        pages = self.extract_pages_from_pdf()

        return pages, metadata

    @staticmethod
    def merge_hyphenated_words(text):
        return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    @staticmethod
    def fix_newlines(text):
        return re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    @staticmethod
    def remove_multiple_newlines(text):
        return re.sub(r"\n{2,}", "\n", text)

    @staticmethod
    def clean_text(pages, cleaning_functions):
        cleaned_pages = []
        for page_num, text in pages:
            for cleaning_function in cleaning_functions:
                text = cleaning_function(text)
            cleaned_pages.append((page_num, text))

        return cleaned_pages

    @staticmethod
    def text_to_docs(text, metadata):
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

    def get_vectorDB(self, return_docs=False):

        document_chunks = self.get_docChunks()
        self.vector_store = FAISS.from_documents(
            document_chunks,
            self.embeddings
        )
        if return_docs:
            return self.vector_store, document_chunks

        return self.vector_store
