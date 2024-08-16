from operator import itemgetter

from pydantic.v1 import SecretStr

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import PyMuPDFLoader

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_community.vectorstores import Qdrant

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, Runnable

from tiktoken import encoding_for_model

from typing import TypedDict

class Query(TypedDict):
	query: str

class Response(TypedDict):
	context: list[Document]
	query: str
	response: AIMessage

def build_chain(openai_api_key: str) -> Runnable[Query, Response]:
	embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
	gpt_3_4_turbo_encoding = encoding_for_model("gpt-3.5-turbo")

	def token_len(text: str) -> int:
		return len(gpt_3_4_turbo_encoding.encode(text))

	taylor_swift_feud = PyMuPDFLoader(
		"https://raw.githubusercontent.com/AI-Maker-Space/DataRepository/main/tswift_fued.pdf"
	).load()

	chunks = RecursiveCharacterTextSplitter(
		chunk_size=200,
		chunk_overlap=50,
		length_function=token_len
	).split_documents(taylor_swift_feud)

	vector_store = Qdrant.from_documents(
		chunks,
		embedding_model,
		location=":memory:",
		collection_name="Taylor Swift - Fued - ADA"
	)

	qdrant_retriever = vector_store.as_retriever()

	rag_prompt = ChatPromptTemplate.from_template(
"""
Answer the question based only on the following context. 
If you cannot answer the question with the context, say that you don't know.

CONTEXT:
{context}

QUERY:
{query}
"""
	)

	openai_client = ChatOpenAI(
		api_key=SecretStr(openai_api_key)
	)

	get_query = itemgetter("query")

	return (
		RunnableParallel(
			{"context": get_query | qdrant_retriever, "query": get_query}
		) | 
		RunnablePassthrough.assign(
			response=rag_prompt | openai_client # type: ignore
		)
	) # type: ignore