from typing import Annotated, Any, Callable

from enum import StrEnum
from operator import itemgetter
# from pprint import pprint

from dotenv import dotenv_values

from pydantic.v1 import SecretStr
from tiktoken import encoding_for_model

from logging import basicConfig, DEBUG

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyMuPDFLoader

from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.output_parsers import StrOutputParser

from langchain_core.tools import BaseTool, tool # pyright: ignore [reportUnknownVariableType]
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough, Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langgraph.prebuilt.chat_agent_executor import AgentState, create_react_agent # pyright: ignore [reportUnknownVariableType, reportMissingTypeStubs]
from langgraph.graph import END, StateGraph # pyright: ignore [reportMissingTypeStubs]


def clean_multi_line(text: str) -> str:
    return "\n".join(
        line.strip() for
        line in
        text
            .strip()
            .split("\n")
    )

def create_vectorstore(documents: list[Document]) -> VectorStoreRetriever:
    def chunk() -> list[Document]:
        gpt_3_4_turbo_encoding = encoding_for_model("gpt-3.5-turbo")

        def token_len(text: str) -> int:
            return len(gpt_3_4_turbo_encoding.encode(text))

        return RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=0,
            length_function=token_len
        ).split_documents(
            documents
        )

    return Qdrant.from_documents(
        chunk(),
        OpenAIEmbeddings(model="text-embedding-3-small"),
        location=":memory:",
        collection_name="llama_3_paper"
    ).as_retriever()

environment_secrets = dotenv_values()

def get_api_key(key_name: str) -> str:
    api_key = environment_secrets[key_name]
    assert api_key != None
    return api_key

llama_3_paper_vectorstore = create_vectorstore(
    PyMuPDFLoader("https://arxiv.org/pdf/2404.19553").load()
)

rag_prompt = ChatPromptTemplate.from_template(
    clean_multi_line(
    """
    CONTEXT:
    {context}

    QUERY:
    {query}

    You are a helpful assistant. 
    Use the available context to answer the question. 
    If you can't answer the question, say you don't know.
    """
    )
)

chat_model = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=SecretStr(get_api_key("OPENAI_API_KEY"))
)

get_query = itemgetter("query")

rag_chain = (
    RunnableParallel(
        {"context": get_query | llama_3_paper_vectorstore, "query": get_query}
    ) |
    rag_prompt |
    chat_model |
    StrOutputParser()
)

rag_chain.invoke(
    {"query": "What does 'context' refer to in 'long context'?"}
)

def create_agent_node[T: AgentState](
    chat_model: ChatOpenAI,
    name: str,
    instructions: str,
    tools: list[BaseTool | Callable[..., Any]],
    state_schema: type[T] = AgentState
):
    agent = create_react_agent(
        chat_model,
        tools, # pyright: ignore [reportArgumentType]
        state_modifier=RunnablePassthrough.assign(
            team_members=lambda state: ", ".join(state["team_members"])
        ) | ChatPromptTemplate.from_messages(  # type: ignore
            [
                ("system",
                    instructions + "\n" + clean_multi_line(
                    """
                    Work autonomously according to your specialty
                    using the tools available to you. 
                    Do not ask for clarification. 

                    Your other team members and other teams will collaborate with you 
                    with their own specialties.

                    You are one of: {team_members}
                    """
                    )
                 ),
                ("placeholder", "{messages}")
            ]
        ),
        state_schema=state_schema
    )

    # def named_agent(state: T):
    #     def state_len(state: dict[str, Any]):
    #         return len(state["messages"])

    #     pre_message_count = state_len(state) # pyright: ignore [reportArgumentType]

    #     new_state = agent.invoke(state)

    #     for message_index in range(pre_message_count, state_len(new_state)):
    #         new_state["messages"][message_index].name = name

    #     return new_state
    
    return agent # RunnableLambda(named_agent)

def bind_agent_node_creator[T: AgentState](
    chat_model: ChatOpenAI,
    state_schema: type[T]
):
    def bound_node(
        name: str,
        instructions: str,
        tools: list[BaseTool | Callable[..., Any]]
    ):
        return create_agent_node(chat_model, name, instructions, tools, state_schema)

    return bound_node

def create_supervisor(
    chat_model: ChatOpenAI,
    instructions: str,
    members: list[str]
) -> Runnable[AgentState, str]:
    NextRole = StrEnum("NextRole", [*members, END])

    @tool
    def select_next_role(
            next_role: Annotated[NextRole, "The next role to select"] # pyright: ignore [reportInvalidTypeForm, reportUnknownParameterType]
    ) -> str: 
        """Selects the next role"""
        return next_role.value # pyright: ignore [reportUnknownMemberType, reportUnknownVariableType]
    
    equipped_chat_model = chat_model.bind_functions([select_next_role]) # pyright: ignore [reportUnknownMemberType]

    return (
        ChatPromptTemplate.from_messages( # pyright: ignore [reportUnknownMemberType]
            [
                ("system", instructions),
                MessagesPlaceholder(variable_name="messages"),
                ("system",
                    clean_multi_line(
                    f"""
                    Is the task complete? If yes, select '{END}'.
                    Otherwise, given the conversation above, who should act next?

                    Select one of the following:
                    {", ".join(f"'{member}'" for member in members)}
                    """
                    )
                ),
            ]
        ) | 
        equipped_chat_model | 
        JsonOutputFunctionsParser() | 
        (lambda tool_request: tool_request["next_role"]) # type: ignore
    )

def empty_node(_: AgentState) -> dict[str, Any]:
    return {"messages": []}

class ResearchTeamState(AgentState):
    team_members: list[str]
    next: str

@tool
def retrieve_information(
    query: Annotated[str, "Query to the Retrieval Augmented Generation tool"]
) -> str:
    """
    Uses a Retrieval Augmented Generation tool to retrieve information about 
    the 'Extending Llama-3’s Context Ten-Fold Overnight' paper
    """
    return rag_chain.invoke({"question": query})

#research_chat_model = ChatOpenAI(model="gpt-4-1106-preview")

create_research_agent = bind_agent_node_creator(
    chat_model,
    ResearchTeamState
)

search_agent = create_research_agent(
    "search",
    "You are a research assistant who can search for up-to-date info using the DuckDuckGo search engine.",
    [DuckDuckGoSearchRun()]
)

paper_agent = create_research_agent(
    "paper_information_retriever",
    "You are a research assistant who can provide specific information on the provided paper: 'Extending Llama-3’s Context Ten-Fold Overnight'",
    [retrieve_information]
)

research_graph = StateGraph(ResearchTeamState)

research_graph.add_node("search", search_agent) # type: ignore
research_graph.add_node("paper_information_retriever", paper_agent) # type: ignore
research_graph.add_node("supervisor", empty_node) # type: ignore

research_graph.add_edge("search", "supervisor")
research_graph.add_edge("paper_information_retriever", "supervisor")

visor = create_supervisor(
    chat_model,
    clean_multi_line(
    """
    You are a supervisor tasked with managing a conversation between the following workers:  
    'search', 'paper_information_retriever'
    
    Given the following user request, respond with the worker to act next. 
    Each worker will perform a task and respond with their results and status.
    """
    ),
    ["search", "paper_information_retriever"],
)

def supervise(state: ResearchTeamState):
    result = visor.invoke(state)
    return result

research_graph.add_conditional_edges(
    "supervisor",
    supervise
)

research_graph.set_entry_point("supervisor")
chain = research_graph.compile()

def main():
    return chain.invoke(
        {
            "messages": [HumanMessage(content=clean_multi_line(
            """
            What are the main takeaways from the paper `Extending Llama-3's Context Ten-Fold Overnight'? 
            Please use 'search' and 'paper_information_retriever'!
            """
            ))],
            "team_members": ["search", "paper_information_retriever"]
        }
    )

from langsmith import Client

langsmith_client = Client(api_key=get_api_key("LANGSMITH_API_KEY"))

from langchain_core.tracers.context import tracing_v2_enabled
from uuid import uuid4

output = None

if __name__ == "__main__":
    basicConfig(
        filename="./logfile",
        filemode="w",
        format="%(levelname)s [%(asctime)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=DEBUG
    )

    with tracing_v2_enabled(project_name=f"Suds Multi Agent Ai - LangGraph - {uuid4().hex[0:8]}", client=langsmith_client):
        output = main()

print(output)