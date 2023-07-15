import json
from dataclasses import dataclass
from typing import Literal

import openai
import streamlit as st

from langchain import LLMChain
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.callbacks import get_openai_callback

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks import StreamlitCallbackHandler

from CustomTools.tools import LookupTool, WebSearchTool, SummarizationTool, ArxivTool


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
        self.tools = self.load_tools()
        self.agent = self.get_agent()

    def load_tools(self):
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature="0",
            openai_api_key=st.session_state['openai_api_key'],
            streaming=False,
            verbose=False
        )

        lookup_tool = LookupTool(llm, st.session_state['vector_store'])
        search_tool = WebSearchTool()
        summarize_tool = SummarizationTool(llm, st.session_state['document_chunks'])
        arxiv_tool = ArxivTool(llm)

        tools = [
            Tool(
                name="Lookup from local database",
                func=lookup_tool.run,
                description="Always useful for finding the exactly written answer to the question by looking "
                            "into a collection of academic documents. Input should be a query, not referencing "
                            "any obscure pronouns from the conversation before that will pull out relevant information "
                            "from the database."
            ),

            Tool(
                name="Search Internet from Arxiv",
                func=arxiv_tool.run,
                description="A wrapper around arxiv.org an Online Research Paper Database. Useful for when you need "
                            "to answer questions about Physics, Mathematics, Computer Science, Quantitative Biology, "
                            "Quantitative Finance, Statistics, Electrical Engineering, and Economics from scientific "
                            "articles on arxiv.org before you perform a regular search on the internet. Input should "
                            "be a search query and not referencing any obscure pronouns from the conversation. "
            ),

            Tool(
                name="Search Internet",
                func=search_tool.run,
                description="Useful when you cannot find a clear answer by looking up the database or from the online "
                            "research paper database so that you need to search the regular internet for general"
                            " web articles for further context and understanding to give an elaborate,exact and "
                            "clear answer. Input should be a fully formed question based on the context of what"
                            "you couldn't find and not referencing any obscure pronouns from the conversation before."
            ),
            Tool(
                name="Summarize Database",
                func=summarize_tool.run,
                description="Only use when the human asks to summarize the entire document. Do not use for any other "
                            "tasks other than the human suggesting you to do so "
            ),
        ]

        return tools

    def get_agent(self):
        chat_model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature="0",
            openai_api_key=st.session_state['openai_api_key'],
            streaming=True,
            verbose=False
        )

        prefix = """You are a Compassionate Teacher Chatbot. Engage in a conversation with a human student about an academic topic using the knowledge you have from a database of documents. 
                    Your goal is to provide answers in the friendliest and most easily understandable manner, making complex subjects relatable to even a 5-year-old child. Utilize examples and 
                    detailed explanations to ensure comprehensive understanding of the topic being discussed. Begin by searching for answers and relevant examples within the database of PDF pages 
                    (documents) provided. If you are unable to find sufficient information, you may consult the online research paper database Arxiv to gather additional knowledge and understanding 
                    from research papers. Only when you still lack the necessary information, you may use a general internet search to find results from web articles. However, always prioritize providing 
                    answers and examples from the database and research papers before resorting to general internet search.
                    You should always provide the final answer as bullet points, for the easier understanding and readability of student.

                    You have access to the following tools: """

        FORMAT_INSTRUCTIONS = """Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the detailed,at most comprehensive result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer based on my observation
        Final Answer: the final answer to the original input question is the full detailed explanation from the Observation provided as bullet points."""

        suffix = """Begin!"

        {chat_history}
        Question: {input}
        {agent_scratchpad}"""

        prompt = ZeroShotAgent.create_prompt(
            self.tools,
            prefix=prefix,
            suffix=suffix,
            format_instructions=FORMAT_INSTRUCTIONS,
            input_variables=["input", "chat_history", "agent_scratchpad"],
        )

        def _handle_error(error) -> str:
            INSTRUCTIONS = """Use the following format:
            
                  Thought: you should always think about what to do
                  Action: the action to take, should be one of [{tool_names}]
                  Action Input: the input to the action  
                  Observation: the detailed, comprehensive result of the action
                  Thought: I now know the final answer based on my observation
                  Final Answer: the final answer to the original input question is the full detailed explanation from the Observation provided as bullet points."""

            ouput = str(error).removeprefix("Could not parse LLM output: `").removesuffix("`")

            response = f"Thought: {ouput}\nThe above completion did not satisfy the Format Instructions given in the Prompt.\nFormat Instructions: {INSTRUCTIONS}\nPlease try again and conform to the format."
            # print("error msg: ", response)
            return response

        memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history")

        llm_chain = LLMChain(llm=chat_model, prompt=prompt)

        agent = ZeroShotAgent(llm_chain=llm_chain, tools=self.tools, verbose=True, handle_parsing_errors=_handle_error)
        agent_chain = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=self.tools, verbose=True, memory=memory, handle_parsing_errors=_handle_error
        )

        return agent_chain

    def run_chain(self):
        st.session_state.doc_sources = []
        st.session_state.google_sources = []

        with st.session_state.chat_placeholder:
            st.chat_message("user").write(st.session_state.human_prompt)
            with st.chat_message('assistant'):
                st_callback = StreamlitCallbackHandler(st.container())
                with get_openai_callback() as cb:
                    llm_response = self.agent(inputs=st.session_state.human_prompt, callbacks=[st_callback])
                    st.session_state.token_count += cb.total_tokens
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
        self.agent = self.get_agent()

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
        # if llm_response is None:
        #     return
        self.store_conversation(llm_response)
        return
