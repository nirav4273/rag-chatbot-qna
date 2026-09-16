from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = SecretStr(os.getenv("OPENAI_API_KEY", ''))
if not openai_api_key:
    raise RuntimeError('Set OPENAI_API_KEY first: $env:OPENAI_API_KEY="your_api_key"')


def load_llm(provider: str):
    if provider == 'openai':
        return ChatOpenAI(
            model=os.getenv('OPENAI_MODEL', '')
        )
    # return ChatGroq(
    #     api_key=groq_api_key,
    #     model_name=os.getenv('GROQ_MODEL', '')
    # )

def init_llm():
    return load_llm(os.getenv('MODEL_PROVIDER', ''))