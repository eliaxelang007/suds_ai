from dotenv import dotenv_values

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document

from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from pydantic.types import SecretStr

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

from pprint import pprint

from json import dumps, load, loads

def get_environment_variable(key_name: str) -> str:
    ENVIRONMENT_SECRETS = "environment_secrets"

    if not hasattr(get_environment_variable, ENVIRONMENT_SECRETS):
        setattr(get_environment_variable, ENVIRONMENT_SECRETS, dotenv_values())

    environment_variables: dict[str, str | None] = getattr(
        get_environment_variable, ENVIRONMENT_SECRETS)

    environment_variable = environment_variables[key_name]
    assert (
        environment_variable is not None
    ), f"Couldn't find environment variable with they key [{key_name}]!"

    return environment_variable

def main() -> None:
    # This variable is unused, but it's how we provide the Open AI API key to langchain.
    _chat_ai = ChatOpenAI(
        api_key=get_environment_variable("OPENAI_API_KEY") #type: ignore
    )

    with open("../data_parsing/parsed.json") as parsed_json_file:
        questions_and_answers = load(parsed_json_file)

    vector_store = QdrantVectorStore.from_documents(
         [
            Document(dumps(question_and_answer)) for 
            question_and_answer in
            questions_and_answers
        ],
        OpenAIEmbeddings(
            model="text-embedding-3-small"
        ),
        location=":memory:"
    ).as_retriever()

    a = vector_store.invoke("What time do you open po?")
    print(a)

if __name__ == "__main__":
    main()