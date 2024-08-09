from openai import AsyncOpenAI
from dotenv import dotenv_values

def main():
    secrets = dotenv_values()

    client = AsyncOpenAI(
        api_key=secrets["OPENAI_API_KEY"]
    )

    


if __name__ == "__main__":
    main()