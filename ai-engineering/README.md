# ai-engineering

Training materials for LLM engineering

# Azure OpenAI

1. Creating an `OpenAI client` using Azure

```python
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key={api_key},
    azure_endpoint={azure_api_endpoint},
    api_version={api_version}
)
```

2. Creating an `assistant` using Azure OpenAI client <br>
   See more [here](https://learn.microsoft.com/en-us/azure/ai-services/openai/assistants-reference?tabs=python).

```python
from openai import AzureOpenAI

assistantClient = AzureOpenAI(
    api_key={api_key},
    azure_endpoint=f"{azure_api_endpoint}/openai/assistants?api-version={api_version}",
    api_version={api_version}
)

assistant = assistantClient.beta.assistants.create(
    name="Math Assistant",
    instructions="You are an AI assistant that can write code to help answer math questions.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4o" # model deployed in Azure
)
```

3. Creating a `thread` using Azure OpenAI client <br>
   See more [here](https://learn.microsoft.com/en-us/azure/ai-services/openai/assistants-reference-threads?tabs=python).

```python
from openai import AzureOpenAI

threadsClient = AzureOpenAI(
    api_key={api_key},
    azure_endpoint=f"{azure_api_endpoint}/openai/threads?api-version={api_version}",
    api_version={api_version}
)

thread = threadsClient.beta.threads.create()
```

4. Creating a `message` using Azure OpenAI client <br>
   See more [here](https://learn.microsoft.com/en-us/azure/ai-services/openai/assistants-reference-messages?tabs=python).

```python
from openai import AzureOpenAI

messageClient = AzureOpenAI(
    api_key={api_key},
    azure_endpoint=f"{azure_api_endpoint}/openai/threads/{thread.id}/messages?api-version={api_version}",
    api_version={api_version}
)

# Add a user question to the thread
message = messageClient.beta.threads.messages.create(
    thread_id={thread.id},
    role="user",
    content="Who are the characters in the story?"
)
```

5. Creating a `run` using Azure OpenAI Client <br>
   See more [here](https://learn.microsoft.com/en-us/azure/ai-services/openai/assistants-reference-runs?tabs=python).

```python
from openai import AzureOpenAI

runClient = AzureOpenAI(
    api_key={api_key},
    azure_endpoint=f"{azure_api_endpoint}/openai/threads/{thread.id}/runs?api-version={api_version}",
    api_version={api_version}
)

# Run the thread
run = runClient.beta.threads.runs.create(
    thread_id={thread.id},
    assistant_id={assistant.id},
)
```

## LangChain with Azure Open AI

1. Create the client

```python
from langchain_openai import AzureChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from operator import itemgetter

client = AzureChatOpenAI(
    model={model}, #eg. gpt-4o
    api_key={api_key},
    api_version={api_version},
    azure_endpoint={azure_endpoint}
)

chain = return (
    {"context": itemgetter("context"), "question": itemgetter("question")}
    | prompt | client | StrOutputParser()
)

response = chain.invoke({"context": {context}, "question": {user_input}})
```

Learn more about Azure OpenAI documentation from [here](https://learn.microsoft.com/en-us/azure/ai-services/openai/).
