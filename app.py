from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv 
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import os



#Load PDF data
def load_pdf_file(data):
    loader = DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
    
    documents=loader.load()

    return documents
            
extracted_data = load_pdf_file(data='../../Data/')

#Split the data in text chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = text_split(extracted_data)


#Download the embeddings import hugging face
def download_huggingface_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

embedding = download_huggingface_embeddings()



#pc = Pinecone(api_key="pcsk_3rfex4_CSnNyXyfH3D82c3R5Tyu2BW638tYMpxhqGnpedexjCeZFmZwuLPgDrmBaiDy8Ee")

index_name = "medicalbot"


#pc.create_index(
#    name=index_name,
#    dimension=384, 
#    metric="cosine", 
#    spec=ServerlessSpec(
#        cloud="aws", 
#        region="us-east-1"
#    ) 
#)

load_dotenv()

os.environ["PINECONE_API_KEY"] =os.getenv("PINECONE_API_KEY") 
os.environ["OPENAI_API_KEY"] =os.getenv("OPENAI_API_KEY") 

print(os.environ["OPENAI_API_KEY"])
#docsearch = PineconeVectorStore.from_documents(documents=text_chunks,embedding=embedding,index_name="medicalbot")

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


llm = OpenAI(temperature=0.4, max_tokens=500)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input": "What is pimple"})
print("Response : ", response["answer"])