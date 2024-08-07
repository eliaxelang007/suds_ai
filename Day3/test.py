from typing import Callable, Any, Protocol, Concatenate
from functools import wraps
from contextlib import contextmanager
from inspect import signature

class OpenAiResource(Protocol):
  @property
  def id(self) -> str: ...

class DeletedOpenAiResource(Protocol):
  @property
  def id(self) -> str: ...

  @property
  def deleted(self) -> bool: ...

  @property
  def object(self) -> str: ...


def ignore_unknown_keyword_args(function: Callable[..., Any]) -> Callable[..., Any]:
  @wraps(function)
  def wrapped(*args: Any, **kwargs: Any):
    parameter_names = set(signature(function).parameters.keys())

    new_arguments = {
      argument_name: argument_value for 
      (argument_name, argument_value) in 
      kwargs.items() if 
      argument_name in parameter_names
    }

    return function(*args, **new_arguments)
  
  return wrapped

@contextmanager
def use_openai_resource(resource: OpenAiResource, deleter: Callable[Concatenate[str, ...], DeletedOpenAiResource]):
  try:
    yield resource
  finally:
    ignore_unknown_keyword_args(deleter)(resource.id, _resource=resource)

from dotenv import dotenv_values
from openai import OpenAI


from openai.types.beta.vector_stores.vector_store_file import VectorStoreFile
from openai.types.beta.vector_stores.vector_store_file_deleted import VectorStoreFileDeleted
#import openai.types.beta.threads.text_content_block as text_content_block
from openai.types.beta.threads.text_content_block import TextContentBlock

from IPython import embed_kernel
from IPython.display import display, Markdown

# WANDB = False

# if WANDB:
#   from wandb.integration.openai import autolog
#   autolog({"project": "First RAG App"})

client = OpenAI(api_key=dotenv_values("../.env")["OPENAI_API_KEY"])
beta = client.beta

def main():
  def vector_store_file_deleter(id: str, _resource: VectorStoreFile) -> VectorStoreFileDeleted:
    return beta.vector_stores.files.delete(id, vector_store_id=_resource.vector_store_id)

  with (
    use_openai_resource(
      client.files.create(
        file=open("./frankenstien.txt", "rb"),
        purpose="assistants"
      ), 
      client.files.delete
    ) as frankenstien_file,
    use_openai_resource(
      beta.vector_stores.create(
          name="Frankenstien Documents"
      ),
      beta.vector_stores.delete
    ) as processed_documents,
    use_openai_resource(
      beta.vector_stores.files.create(
        vector_store_id=processed_documents.id,
        file_id=frankenstien_file.id
      ),
      vector_store_file_deleter
    ),
    use_openai_resource(
      beta.assistants.create(
          name="Frank(enstien)",
          instructions="""
      You are a librarian who'd like to answer questions about the book Frankenstien by Mary Shelley.
      The book is included in a file for your reference; 
      when you answer, you cite your source in the book and explain why you're correct.
      If you don't know the answer, say so.
      """,
          model="gpt-3.5-turbo",
          tools=[{"type": "file_search"}],
          tool_resources={"file_search": {"vector_store_ids": [processed_documents.id]}}
      ),
      beta.assistants.delete
    ) as assistant,
    use_openai_resource(
      beta.threads.create(),
      beta.threads.delete
    ) as thread
  ):  
    _message = beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"What is the first words Victor Frankenstein speaks?"
    )

    run = beta.threads.runs.create(
      thread_id=thread.id,
      assistant_id=assistant.id,
    )

    while run.status == "in_progress" or run.status == "queued":
      run = beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
      )

    messages = beta.threads.messages.list(
      thread_id=thread.id
    )

    for message in messages:
      for content_chunk in message.content:
        match content_chunk:
          case TextContentBlock() as text_content:
            display(Markdown(text_content.text.value))
          
          case _:
            display(Markdown("### Unsupported content chunk!"))


if __name__ == "__main__":
    main()