
from dotenv import load_dotenv
# from llm import init_llm
# from langchain_community.agent_toolkits import SQLDatabaseToolkit
# from langchain.agents import create_agent
# from langgraph.checkpoint.memory import InMemorySaver
# import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from llm import init_llm
from langchain.tools import tool

load_dotenv()


model = init_llm()

assert model is not None, "init_llm() returned None"

# memory = InMemorySaver()
memory = InMemorySaver()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

# ### Load PDF
# loader = PyPDFLoader("./files/sample1.pdf")
# result = loader.load()

# ## Text Splitter
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500, 
#     chunk_overlap=150
# )
# splitter = text_splitter.split_documents(result)


# vector_store = Chroma.from_documents(
#     documents=splitter,
#     embedding=embeddings,
#     persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
# )


vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)


prompt = ChatPromptTemplate.from_template(
    """
    You are assistant who provide ans based on the given context and question.
    Need to use only provided context and if not data found mention -> No data found
    Context: {content}
    Question: {question}
    """
)

@tool
def retrival_tool(query: str):
    """
        This tool help you to retrive the data of the provided PDF document.
    """
    print("Tool call>>", query)
    search_result = vector_store.similarity_search(query, k=3)
    
    content = ''

    for record in search_result:
        content += record.page_content

    return content



agent = create_agent(
    model=model,
    # checkpointer=memory,
    tools=[retrival_tool],
    system_prompt="You are RAG assistant who provide ans from the provided document context ONLY and use the retrival_tool for the fetching data and when have multiple questions return answers in better formatting."
)


# rendered_prompt = prompt.format(content=content, question=query)

result = agent.invoke({
    "messages": [
        {
            'role': 'user',
            'content': """
                What are the findings or conclusions presented?
                Are there any important statistics or data mentioned?
            """
        }
    ]
})
messages = result['messages']

print("ANS ---> ", messages[-1].content)

# chain = RunnableSequence(prompt, model)









