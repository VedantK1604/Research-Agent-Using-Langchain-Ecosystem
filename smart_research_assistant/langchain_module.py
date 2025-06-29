import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, SecretStr


import tempfile

load_dotenv()

api = os.getenv("OPENROUTER_API_KEY")


class ResearchAssistantLangchain:

    def __init__(self):
        """Initialize the Research Assistant with necessary configurations."""

        # self.llm = ChatOllama(model="gemma3n:e4b")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        self.vectordb = None
        self.persist_directory = tempfile.mkdtemp()
        print(f"Created temporary directory: {self.persist_directory}")

    def load_urls(self, urls: List[str]) -> List[Document]:
        """Implement the logic to load documents from URLs"""

        all_documents = []

        for url in urls:
            try:
                loader = WebBaseLoader(url)
                data = loader.load()

                split_document = self.text_splitter.split_documents(data)
                all_documents.extend(split_document)

                print(f"successfully loaded and processed {url}")
            except Exception as e:
                print(f"Error loading {url}: {e}")

        return all_documents

    def create_vector_stores(self, documents: List[Document]):

        try:
            self.vector_db = FAISS.from_documents(
                documents=documents, embedding=self.embeddings
            )
            print(f"Create FAISS vector store with {len(documents)} documents.")
        except Exception as e:
            print(f"Error creating FAISS vector store: {e}")
            try:
                self.vector_db = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=None,
                )
                print(f"Create Chroma vector store with {len(documents)} documents.")
            except Exception as e2:
                print(f"Error creating Chroma vector store: {e2}")
                raise ValueError(f"Failed to create vector store: {e2}")

    def query_data(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """
        Queries the vector store using ConversationalRetrievalChain with a custom prompt,
        returning the answer and source documents.
        """

        if not self.vector_db:
            raise ValueError(
                "Vector store not initialized, please load documents first."
            )

        retriever = self.vector_db.as_retriever(search_kwargs={"k": num_results})

        prompt_text = """
        Answer the following question based on the context provided.

        <context>
        {context}
        </context>

        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(prompt_text)
        document_chain = create_stuff_documents_chain(self.llm, prompt)

        retriever_chain = ConversationalRetrievalChain(
            retriever=retriever,
            combine_docs_chain=document_chain,
            llm=self.llm,
        )

        # ✅ FIXED: Add empty chat_history to prevent infinite loop
        response = retriever_chain.invoke(
            {
                "question": query,
                "chat_history": [],  # Prevents hanging/looping by ensuring stateless behavior
            }
        )

        return {
            "answer": response["answer"],
            "source_documents": response["source_documents"],
        }

    def summarize_document(self, document: str) -> str:

        message = HumanMessage(
            content=f"Summarize the following document in a concise but comprehensive manner: \n\n {document}"
        )
        response = self.llm.invoke([message])
        # Ensure the return value is always a string
        if isinstance(response.content, str):
            return response.content
        elif isinstance(response.content, list):
            return "\n".join(str(item) for item in response.content)
        else:
            return str(response.content)
