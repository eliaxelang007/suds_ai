from typing import Literal, ClassVar
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, InitVar
from enum import Enum

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionFunctionMessageParam,
  )

from openai.types.chat.completion_create_params import ResponseFormat

from openai.types.beta.assistant import Assistant as OpenaiAssistant
from openai.types.beta.assistant_tool_param import AssistantToolParam
from openai.types.beta.file_search_tool_param import FileSearchToolParam
from openai.types.beta.code_interpreter_tool_param import CodeInterpreterToolParam
from openai.types.beta.function_tool_param import FunctionToolParam

from openai._types import NotGiven

from dotenv import dotenv_values

# WANDB = False

# if WANDB:
#   from wandb.integration.openai import autolog
#   autolog({"project": "First RAG App"})

openai = OpenAI(api_key=dotenv_values("../.env")["OPENAI_API_KEY"])

RawConversationMessage = ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam | ChatCompletionAssistantMessageParam | ChatCompletionToolMessageParam | ChatCompletionFunctionMessageParam

class Role(Enum):
  System = "system"
  User = "user"
  Assistant = "assistant"
  Tool = "tool"

class AiModel(Enum):
    Gpt4O = "gpt-4o"
    Gpt4O_2024_05_13 = "gpt-4o-2024-05-13"
    Gpt4O_Mini = "gpt-4o-mini"
    Gpt4O_Mini_2024_07_18 = "gpt-4o-mini-2024-07-18"
    Gpt4_Turbo = "gpt-4-turbo"
    Gpt4_Turbo_2024_04_09 = "gpt-4-turbo-2024-04-09"
    Gpt4_0125_Preview = "gpt-4-0125-preview"
    Gpt4_Turbo_Preview = "gpt-4-turbo-preview"
    Gpt4_1106_Preview = "gpt-4-1106-preview"
    Gpt4_Vision_Preview = "gpt-4-vision-preview"
    Gpt4 = "gpt-4"
    Gpt4_0314 = "gpt-4-0314"
    Gpt4_0613 = "gpt-4-0613"
    Gpt4_32K = "gpt-4-32k"
    Gpt4_32K_0314 = "gpt-4-32k-0314"
    Gpt4_32K_0613 = "gpt-4-32k-0613"
    Gpt3_5_Turbo = "gpt-3.5-turbo"
    Gpt3_5_Turbo_16K = "gpt-3.5-turbo-16k"
    Gpt3_5_Turbo_0301 = "gpt-3.5-turbo-0301"
    Gpt3_5_Turbo_0613 = "gpt-3.5-turbo-0613"
    Gpt3_5_Turbo_1106 = "gpt-3.5-turbo-1106"
    Gpt3_5_Turbo_0125 = "gpt-3.5-turbo-0125"
    Gpt3_5_Turbo_16K_0613 = "gpt-3.5-turbo-16k-0613"

@dataclass(frozen=True, slots=True)
class AssistantTool(ABC):
    @staticmethod
    @abstractmethod
    def type() -> str:
       ...

    @abstractmethod
    def into(self) -> AssistantToolParam:
       ...

@dataclass(frozen=True, slots=True)
class CodeInterpreter(AssistantTool):
    @staticmethod
    def type() -> Literal["code_interpreter"]:
       return "code_interpreter"

    def into(self) -> CodeInterpreterToolParam:
       return {"type": self.type()}

@dataclass(frozen=True, slots=True)
class FileSearch(AssistantTool):
    max_result_count: int

    @staticmethod
    def type() -> Literal["file_search"]:
       return "file_search"
    
    def into(self) -> FileSearchToolParam:
       return {"type": self.type(), "file_search": {"max_num_results": self.max_result_count}}

@dataclass(frozen=True, slots=True)
class Function(AssistantTool):
    name: str
    description: str
    parameters: dict[str, object]

    @staticmethod
    def type() -> Literal["function"]:
       return "function"
    
    def into(self) -> FunctionToolParam:
       return {
          "type": self.type(), 
          "function": {
             "name": self.name, 
             "description": self.description, 
             "parameters": self.parameters
        }
    }


# class AssistantTool(Enum):
#    CodeInterpreter = "code_interpreter"
#    FileSearch = "file_search"
#    Function = "function"

@dataclass(frozen=True, order=True, slots=True)
class Message():
  role: Role
  content: str

  def into(self) -> RawConversationMessage:
    return {"role": self.role.value, "content": self.content} # type:ignore

  @classmethod
  def from_raw(cls, raw: ChatCompletionMessage) -> "Message":
    return cls(Role(raw.role), raw.content or "")

def system(message: str) -> Message:
  return Message(Role.System, message)

def user(message: str) -> Message:
  return Message(Role.User, message)

def respond(messages: list[Message], response_format: ResponseFormat | NotGiven=NotGiven(), model: AiModel = AiModel.Gpt3_5_Turbo) -> Message:
  raw_messages = [message.into() for message in messages]

  return Message.from_raw(
    openai.chat.completions.create(
      model=model.value,
      messages=raw_messages,
      response_format=response_format
    ).choices[0].message
  )

@dataclass(slots=True, frozen=True)
class Chatbot:
    model: AiModel = AiModel.Gpt3_5_Turbo
    messages: list[Message] = field(default_factory=list)

    def respond(self, response_format: ResponseFormat | NotGiven=NotGiven()) -> Message:
        response = respond(self.messages, response_format = response_format, model=self.model)
        self.messages.append(response)
        return response

    def converse(self, chat: Message, response_format: ResponseFormat | NotGiven=NotGiven()) -> Message:
        self.messages.append(chat)
        return self.respond(response_format=response_format)

@dataclass(slots=True, frozen=True)
class Assistant():
    __create_key: ClassVar[object] = object()

    create_key: InitVar[object]
    openai_assistant: OpenaiAssistant = field(init=False)
    model: InitVar[AiModel]
    name: InitVar[str]
    description: InitVar[str]
    instructions: InitVar[str]
    tools: InitVar[list[AssistantTool]] = field(default_factory=list)

    def __post_init__(self, create_key: object, model: AiModel, name: str, description: str, instructions: str, tools: list[AssistantTool]):
       assert create_key == type(self).__create_key, "Don't create this class manually; create this class through an instance of [Thread] instead!"

       object.__setattr__(self, "openai_assistant", openai.beta.assistants.create(
          model=model.name,
          name=name,
          description=description,
          instructions=instructions,
          tools=[tool.into() for tool in tools]
       ))

@dataclass(slots=True, frozen=True)
class Thread():
   