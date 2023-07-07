from abc import ABC, abstractmethod
import requests
import justext

import streamlit as st

from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from Serp.serp import CustomSerpAPIWrapper
from langchain.agents import load_tools


class CustomTool(ABC):
    @abstractmethod
    def run(self, query: str):
        pass


class SummarizationTool(CustomTool):
    def __init__(self, llm, document_chunks):
        self.chain = load_summarize_chain(llm, chain_type="map_reduce")
        self.document_chunks = document_chunks

    def run(self, query: str):
        return self.run_chain()

    @st.cache
    def run_chain(self):
        return self.chain.run(self.document_chunks)


class LookupTool(CustomTool):
    def __init__(self, llm, vector_store):
        self.retrieval = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(),
            return_source_documents=True
        )

    def run(self, query: str):
        res = self.retrieval(query)
        st.session_state.doc_sources = res['source_documents']
        return res['result']


class ArxivTool(CustomTool):
    def __init__(self, llm):
        self.arxiv = load_tools(["arxiv"], llm=llm)[0]

    def run(self, query: str):
        response = self.arxiv.run(query)
        st.session_state.google_sources.append(response)
        return response


class WebSearchTool(CustomTool):
    def __init__(self):
        self.search = CustomSerpAPIWrapper(serpapi_api_key=st.secrets['serp_api_key'])
        self.answer = ""
        self.source_dict = {}
        self.title = ""
        self.link = ""
        self.snippet = ""
        self.idx = 0

    def get_title(self):
        if "title" in self.source_dict['organic_results'][self.idx]:
            return self.source_dict['organic_results'][self.idx]['title']

    def get_link(self):
        if "link" in self.source_dict['organic_results'][self.idx]:
            return self.source_dict['organic_results'][self.idx]['link']

    def get_snippet(self):
        if "snippet" in self.source_dict['organic_results'][self.idx]:
            return self.source_dict['organic_results'][self.idx]['snippet']

    def get_paragraphs(self, paragraphs):

        relevant_paragraphs = []
        for paragraph in paragraphs:
            if paragraph.class_type == "good":
                relevant_paragraphs.append(paragraph)

        answer = ""

        # Define the number of paragraphs to consider from each section
        num_paragraphs_beginning = int(len(relevant_paragraphs) * 0.40)
        num_paragraphs_middle = int(len(relevant_paragraphs) * 0.50)
        num_paragraphs_end = int(len(relevant_paragraphs) * 0.10)

        # Extract paragraphs from the beginning
        words_count = 0
        for i in range(min(num_paragraphs_beginning, len(relevant_paragraphs))):
            paragraph = relevant_paragraphs[i]
            if paragraph.words_count > 40:
                if words_count < 50:
                    # print(f"Got from beg : {paragraph.text}\n\n")
                    answer += paragraph.text + "\n"
                    words_count += paragraph.words_count

        # Extract paragraphs from the middle
        words_count = 0
        middle_start = len(relevant_paragraphs) // 2 - num_paragraphs_middle // 2
        middle_end = middle_start + num_paragraphs_middle

        for i in range(middle_start, middle_end):
            if 0 <= i < len(relevant_paragraphs):
                paragraph = relevant_paragraphs[i]
                if paragraph.words_count > 10:
                    if words_count < 50:
                        # print(f"Got from mid : {paragraph.text}\n\n")
                        answer += paragraph.text + "\n"
                        words_count += paragraph.words_count

        # Extract paragraphs from the end
        words_count = 0
        end_start = len(relevant_paragraphs) - num_paragraphs_end
        end_end = len(relevant_paragraphs)

        for i in range(end_start, end_end):
            if 0 <= i < len(relevant_paragraphs):
                paragraph = relevant_paragraphs[i]
                if paragraph.words_count > 20:
                    if words_count < 50:
                        # print(f"Got from end : {paragraph.text}\n\n")
                        answer += paragraph.text + "\n"
                        words_count += paragraph.words_count

        return answer

    def get_answer(self):

        answer = "No Text Scraped"
        try:
            response = requests.get(self.link)
            paragraphs = justext.justext(response.content, justext.get_stoplist("English"))

        except Exception as e:
            return answer

        answer = self.get_paragraphs(paragraphs)

        return answer

    def run(self, query: str):

        result = self.search.run(query)
        self.source_dict = result['source_dict']
        self.idx = result['idx']

        self.title = self.get_title()
        self.link = self.get_link()
        self.snippet = self.get_snippet()
        self.answer = self.get_answer()
        st.session_state.google_sources.append({"title": self.title, "link": self.link,
                                                "snippet": self.snippet, "answer": self.answer})

        return self.answer
