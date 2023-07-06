import json
from dataclasses import dataclass
from typing import Literal

import openai
import streamlit as st

from langchain import LLMChain
from langchain.agents import Tool
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.callbacks import get_openai_callback

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks import StreamlitCallbackHandler
from langchain.schema.output_parser import OutputParserException

from CustomTools.tools import LookupTool, WebSearchTool, SummarizationTool


@dataclass
class Message:
    """Class for keeping track of a chat message."""
    role: Literal["human", "ai"]
    content: str


class ConversationalAgent:
    """ Conversational Agent Class.

        This class is used to create a Conversational Agent, which can be used to chat with a user.
        The agent is initialized with a StreamHandler, which is used to stream the output of the agent
        to a streamlit container.

        Handles the following:
            - Creating necessary tools and agent for searching answers in PDF and the Web.
            - Running the agent chain on a query.
            - Storing memory and source documents.
            - Clearing the conversation
            - Curating key phrases from the conversation.

    """

    def __init__(self):
        self.agent = self.get_agent_chain()

    def get_agent_chain(self):

        chat_model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature="0",
            openai_api_key=st.session_state['openai_api_key'],
            streaming=True,
            verbose=False
        )

        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature="0",
            openai_api_key=st.session_state['openai_api_key'],
            streaming=False,
            verbose=False
        )

        lookup_tool = LookupTool(chat_model, st.session_state['vector_store'])
        search_tool = WebSearchTool()
        summarize_tool = SummarizationTool(llm, st.session_state['document_chunks'])

        tools = [
            Tool(
                name="Lookup from database",
                func=lookup_tool.run,
                description="Always useful for finding the exactly written answer to the question by looking into a collection of academic"
                            "documents. Input should be a query, not referencing any obscure pronouns "
                            "from the conversation before that will pull out relevant information from the database."
            ),

            Tool(
                name="Search Internet",
                func=search_tool.run,
                description="Unless you cannot find a clear answer by looking up the database you need to "
                            "search the internet for further context and understanding to give a elaborate, "
                            "exact and clear answer. Input should be a fully formed question based on the context of "
                            "what you couldn't find and not referencing any obscure pronouns from the conversation "
                            "before. "
            ),
            Tool(
                name="Summarize Database",
                func=summarize_tool.run,
                description="Only use when the human asks to summarize the entire document. Do not use for any other "
                            "tasks other than the human suggesting you to do so "
            ),
        ]

        prefix = """Have a conversation with a human over an academic topic from a database of documents, 
        answering the following questions as best, academically and elaborately as you can. 
        Your goal is to provide as much detail as you can possibly gather 
        from the database of documents by thoroughly going through it.
        If you get a proper exact answer from the 
        document, do not try to summarise what you have observed and understood, give your final answer as the 
        observations and answers you found, exactly as it is. 
        Only when you cannot gather enough information from the 
        document, to gather in-depth knowledge,you may take help of the internet search. 
        You want the human to keep asking insightful questions, that can help them learn more about the subject. 
            
        You have access to the following tools: """

        suffix = """Begin!"

        {chat_history}
        Question: {input}
        {agent_scratchpad}"""

        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=["input", "chat_history", "agent_scratchpad"],
        )
        memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history")

        llm_chain = LLMChain(llm=chat_model, prompt=prompt)

        agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True,
                              handle_parsing_errors=True)
        agent_chain = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True, memory=memory
        )

        return agent_chain

    def run_chain(self):

        st.session_state.doc_sources = []
        st.session_state.google_sources = []

        with get_openai_callback() as cb:
            st.session_state.token_count += cb.total_tokens

            with st.session_state.chat_placeholder:
                st.chat_message("user").write(st.session_state.human_prompt)
                with st.chat_message('assistant'):
                    try:
                        st_callback = StreamlitCallbackHandler(st.container())
                        llm_response = self.agent(st.session_state.human_prompt, callbacks=[st_callback])
                    except OutputParserException as e:
                        response = str(e)
                        response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
                        llm_response = {"output": response, "chat_history": ""}
                        st.error(e)

            return llm_response

    def regenerate_response(self):

        st.session_state.human_prompt = st.session_state.history[-2].content
        st.session_state.history = st.session_state.history[:-2]
        self.run_callback()
        return

    def clear_conversation(self):

        st.session_state.history = []
        st.session_state.search_keywords = []
        st.session_state.sources = []
        st.session_state.doc_sources = []
        st.session_state.google_sources = []
        self.agent = self.get_agent_chain()

    def store_conversation(self, llm_response):

        st.session_state.search_keywords += self.get_keywords(llm_response)

        st.session_state.history.append(
            Message("human", st.session_state.human_prompt)
        )
        st.session_state.history.append(
            Message("ai", llm_response["output"])
        )

        st.session_state.human_prompt = ""

        st.session_state.search_keywords = st.session_state.search_keywords[-5:]

    @staticmethod
    def get_keywords(llm_response):

        conversation = llm_response["chat_history"]
        keyword_list = []

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
        if llm_response is None:
            return
        self.store_conversation(llm_response)
        return
