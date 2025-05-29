import os

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich import print
from dotenv import load_dotenv



load_dotenv()

token = os.getenv("SECRET")

model = "gpt-4.1-nano"

    

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model=model, api_key=token )


# Load, chunk and index the contents of the blog.

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2017-06-21-overview/",),
        bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)


vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(
    model="text-embedding-3-small",
    
    api_key=token,
))

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

print(prompt)


def format_docs(docs):
    print(docs)
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


rag_chain.invoke("What is Convolutional Neural Networks?")






