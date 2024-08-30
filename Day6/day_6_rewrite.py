import stat
from typing import Annotated, Any, Callable, cast

from enum import StrEnum
# from pprint import pprint

from dotenv import dotenv_values
from uuid import uuid4

from pydantic.v1 import SecretStr
from tiktoken import encoding_for_model

from logging import basicConfig, DEBUG

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.tools import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyMuPDFLoader

from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.output_parsers import StrOutputParser

# pyright: ignore[reportUnknownVariableType]
from langchain_core.tools import BaseTool, tool
# pyright: ignore[reportUnknownVariableType]
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.tracers.context import tracing_v2_enabled #pyright: ignore[reportUnknownVariableType]
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough, Runnable
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs]
from langgraph.prebuilt.chat_agent_executor import AgentState, create_react_agent
# pyright: ignore[reportMissingTypeStubs]
from langgraph.graph import END, StateGraph

from langsmith import Client


def clean_multi_line(text: str) -> str:
    return "\n".join(
        line.strip() for
        line in
        text
            .strip()
            .split("\n")
    )


def create_vectorstore(documents: list[Document]) -> VectorStoreRetriever:
    gpt_3_4_turbo_encoding = encoding_for_model("gpt-3.5-turbo")

    def token_len(text: str) -> int:
        return len(gpt_3_4_turbo_encoding.encode(text))

    return Qdrant.from_documents(
        RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=0,
            length_function=token_len
        ).split_documents(
            documents
        ),
        OpenAIEmbeddings(model="text-embedding-3-small"),
        location=":memory:",
        collection_name="llama_3_paper"
    ).as_retriever()


def get_api_key(key_name: str) -> str:
    ENVIRONMENT_SECRETS = "environment_secrets"

    if not hasattr(get_api_key, ENVIRONMENT_SECRETS):
        setattr(get_api_key, ENVIRONMENT_SECRETS, dotenv_values())

    environment_secrets: dict[str, str | None] = getattr(
        get_api_key, ENVIRONMENT_SECRETS)

    api_key = environment_secrets[key_name]
    assert api_key != None

    return api_key


def create_rag_chain(chat_ai: ChatOpenAI, document: list[Document]) -> Runnable[str, str]:
    return (
        {
            "context": RunnablePassthrough() | create_vectorstore(document), 
            "query": RunnablePassthrough()
        } |
        ChatPromptTemplate.from_template(
            clean_multi_line(
                """
                CONTEXT:
                {context}

                QUERY:
                {query}

                You are a helpful assistant. 
                Use the available context to answer the question. 
                If the answer to the question is not in the context,
                or you are unable to answer the question, say you don't know.
                """
            )
        ) |
        chat_ai |
        StrOutputParser()
    )  # pyright: ignore[reportUnknownVariableType]

def create_agent_node[T: AgentState](
    chat_ai: ChatOpenAI,
    name: str,
    instructions: str,
    tools: list[BaseTool],
    state_schema: type[T] = AgentState
) -> Runnable[T, T]:
    def name_messages(state: AgentState) -> list[BaseMessage]:
        return [
            type(message)(**message.dict(exclude={"name"}), name=name) for 
            message in 
            state["messages"]
        ]

    return create_react_agent(
        chat_ai,
        tools,
        state_modifier=instructions + clean_multi_line(
            """
            Work autonomously according to your specialty using the tools available to you. 
            Do not ask for clarification. 
            """
        ),
        state_schema=state_schema
    ) | RunnablePassthrough.assign(
        messages=name_messages
    ) # pyright: ignore[reportReturnType]

def main():
    CHAT_MODEL = "gpt-3.5-turbo"

    chat_ai = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=SecretStr(get_api_key("OPENAI_API_KEY"))
    )

    llama_3_paper_rag_chain = create_rag_chain(
        chat_ai,
        PyMuPDFLoader("https://arxiv.org/pdf/2404.19553").load()
    )

    searcher_agent = create_agent_node(
        chat_ai,
        "searcher_agent",
        "You are a research assistant who can search for up-to-date info using the Tavily search engine.",
        [TavilySearchResults(
            api_wrapper=TavilySearchAPIWrapper(tavily_api_key=SecretStr(
                get_api_key("TAVILY_API_KEY")
            ))
        )],
    )

    #print
    (
        searcher_agent.invoke(
            {
                "messages" : [
                    HumanMessage(
                        "Could you give me a the main points of the 'Extending Llama-3's Context Ten-Fold Overnight' paper?"
                    )
                ], 
                "is_last_step": False
            }
        )
    )

def trace():
    with tracing_v2_enabled(
        project_name=f"Suds Multi Agent Ai - LangGraph - {uuid4().hex[0:8]}",
        client=Client(api_key=get_api_key("LANGSMITH_API_KEY"))
    ):
        return main()


if __name__ == "__main__":
    trace()