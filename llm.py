from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from pydantic import SecretStr
import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = SecretStr(os.getenv("GROQ_API_KEY", ''))
print(groq_api_key)
if not groq_api_key:
    raise RuntimeError('Set GROQ_API_KEY first: $env:GROQ_API_KEY="your_api_key"')

openai_api_key = SecretStr(os.getenv("OPENAI_API_KEY", ''))
if not openai_api_key:
    raise RuntimeError('Set OPENAI_API_KEY first: $env:OPENAI_API_KEY="your_api_key"')


def load_llm(provider: str):
    if provider == 'openai':
        return ChatOpenAI(
            model=os.getenv('OPENAI_MODEL', '')
        )
    return ChatGroq(
        api_key=groq_api_key,
        model=os.getenv('GROQ_MODEL', '')
    )

def init_llm():
    return load_llm(os.getenv('MODEL_PROVIDER', ''))