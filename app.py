import chainlit as cl
import openai
from openai import AzureOpenAI
from pathlib import Path
from typing import List
from langchain_openai import AzureOpenAIEmbeddings,AzureChatOpenAI
from dotenv import load_dotenv 
from langchain.schema import Document
from langchain_pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader,
)
from pinecone import Pinecone, ServerlessSpec
import uuid
# import phoenix as px
# px.launch_app().view()
# from phoenix.otel import register

# tracer_provider = register()

PDF_STORAGE_PATH = "./pdfs"

PINECONE_API_KEY= '6e1c77c5-ea85-46c3-85c4-864606a6c364'

index_name = 'primer'
# pc = Pinecone(api_key=PINECONE_API_KEY)
# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name,
#         dimension=1536,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws",region="us-west-2") 
#     )

embeddings = AzureOpenAIEmbeddings(
deployment="llmops_CT_textembedding",
model="llmops_CT_textembedding",
azure_endpoint="https://llmops-classroom-openai.openai.azure.com/",
openai_api_key="4784a65517d94ebc9c1753edbcefe69b",
openai_api_version="2023-07-01-preview"
)

# def process_pdfs(pdf_storage_path: str):
#     pdf_directory = Path(pdf_storage_path)
#     docs = []
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
#     # Load PDFs and split into documents
#     for pdf_path in pdf_directory.glob("*.pdf"):
#         loader = PyMuPDFLoader(str(pdf_path))
#         documents = loader.load()
#         docs += text_splitter.split_documents(documents)
    

#     # Convert text to embeddings
#     for doc in docs:
#         embedding = embeddings.embed_query(doc.page_content)
#         random_id = str(uuid.uuid4())
#         #print (embedding)
#         doc_search = pc.Index(index_name)
#         #doc_search = Pinecone (doc_search, embeddings.embed_query, doc.page_content, random_id)
    
#     # Store the vector in Pinecone index
#         doc_search.upsert(vectors = [{"id": random_id, "values": embedding, "metadata": {"source": doc.page_content}}])
#         print("Vector stored in Pinecone index successfully.")
#     return doc_search

# doc_search = process_pdfs(PDF_STORAGE_PATH)


from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.pinecone import Pinecone
import pinecone
namespace=None
welcome_message="Hi. This is the beginning of the chat."

# @cl.on_chat_start
# async def start():
#     await cl.Message(content=welcome_message).send()
#     docsearch = Pinecone.from_existing_index(
#         index_name=index_name, embedding=embeddings, namespace=namespace
#     )

#     message_history = ChatMessageHistory()

#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         output_key="answer",
#         chat_memory=message_history,
#         return_messages=True,
#     )

#     chain = ConversationalRetrievalChain.from_llm(
#         llm = AzureChatOpenAI(
#         api_key="4784a65517d94ebc9c1753edbcefe69b",
#         azure_endpoint="https://llmops-classroom-openai.openai.azure.com/",
#         api_version="2023-07-01-preview",
#         openai_api_type="azure",
#         azure_deployment="llmops_CT_GPT4o",
#         streaming=True
#         ),
#         chain_type="stuff",
#         retriever=docsearch.as_retriever(),
#         memory=memory,
#         return_source_documents=True,
#     )
#     cl.user_session.set("chain", chain)




# @cl.on_message
# async def main(message: cl.Message):
#     chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain

#     cb = cl.AsyncLangchainCallbackHandler()

#     res = await chain.acall(message.content, callbacks=[cb])
#     answer = res["answer"]
#     source_documents = res["source_documents"]  # type: List[Document]

#     text_elements = []  # type: List[cl.Text]

#     if source_documents:
#         for source_idx, source_doc in enumerate(source_documents):
#             source_name = f"source_{source_idx}"
#             # Create the text element referenced in the message
#             text_elements.append(
#                 cl.Text(content=source_doc.page_content, name=source_name)
#             )
#         source_names = [text_el.name for text_el in text_elements]

#         if source_names:
#             answer += f"\nSources: {', '.join(source_names)}"
#         else:
#             answer += "\nNo sources found"

#     await cl.Message(content=answer, elements=text_elements).send()   

@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...
    def call_LLM(PromptTemplate):
     
     client = AzureOpenAI(
       azure_endpoint = "https://llmops-classroom-openai.openai.azure.com/", 
       api_key= "4784a65517d94ebc9c1753edbcefe69b",  
       api_version="2023-07-01-preview"
     )


     response = client.chat.completions.create(
      temperature=0.1,
      model="llmops_CT_GPT4o", # model = "deployment_name".
      messages=[
          {"role": "user", "content": PromptTemplate},
        ])

  
     return response.choices[0].message.content
    
    response_text= call_LLM(message.content)

    # Send a response back to the user
    await cl.Message(
        content=response_text,
    ).send()
