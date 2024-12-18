from typing import Any
from dotenv import dotenv_values

from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from langchain_qdrant import QdrantVectorStore

from operator import itemgetter

from json import dumps, load

def clean_multi_line(text: str) -> str:
    return "\n".join(
        line.strip() for
        line in
        text
            .strip()
            .split("\n")
    )

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
    chat_ai = ChatOpenAI(
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

    rag_prompt = ChatPromptTemplate.from_template(
        clean_multi_line(
        """
        CONTEXT: ```
        {context}
        ```

        QUERY: ```
        {query}
        ```
        
        You are a helpful assistant that's answering questions for a laundry shop.
        The CONTEXT given to you are previous questions asked by customers and their
        responses by the store employee; use these Q&As to answer the QUERY.
        Speak as if you're an employee yourself and in the employees' response style.

        If you don't know the answer, tell the customer to refer to the store employee instead.

        NEVER UNDER ANY CIRCUMSTANCE GIVE ANY KIND OF PRICE TO THE CUSTOMER.
        FOR PRICE QUESTIONS, ALWAYS DEFER TO THE STORE EMPLOYEE.
        """
        )
    )

    get_query = itemgetter("query") 

    rag_chain = (
        RunnableParallel(
            {"context": get_query | vector_store, "query": get_query}
        ) |
        rag_prompt |
        chat_ai
    )

    user_input = input("[Customer]: ")
    response = rag_chain.invoke({"query": user_input})

    print(f"[Chatbot]: {response.content}")


if __name__ == "__main__":
    main()