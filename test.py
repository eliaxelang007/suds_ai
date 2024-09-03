from typing import Literal, Union, Annotated
from mini_langchain import to_openai_tool
from json import dumps

def retrieve_information(
    a: Annotated[str | None, "Hello!"], 
    b: Annotated[Literal["1"] | Literal["2"], "I'm"], 
    c: Annotated[Literal["a", "d"], "scruffy"]
) -> str:
    """Retrieve information"""
    return "i am information"

as_tool = to_openai_tool(retrieve_information)

print(as_tool)

print(dumps(as_tool, indent=4))
