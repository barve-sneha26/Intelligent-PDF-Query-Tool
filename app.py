import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# Initialize Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Hugging Face Hub LLM
llm = HuggingFaceHub(
    repo_id="facebook/bart-large-cnn",  # Hugging Face model for summarization
    model_kwargs={"temperature": 0, "max_length": 200}
)

# Streamlit app setup
st.title("PDF Query Application")
st.write("Upload a PDF file, query its content, and get responses.")

# File upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save uploaded file temporarily
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Extract text from the PDF
    def extract_text_from_pdf(file_path):
        pdf_reader = PdfReader(file_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    # Load and process the PDF text
    text = extract_text_from_pdf(temp_file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = splitter.create_documents([text])

    # Create FAISS vector store
    vectordb = FAISS.from_documents(documents, embeddings)
    retriever = vectordb.as_retriever(search_k=5)

    # Query input
    st.write("### Query the PDF")
    query = st.text_input("Enter your query:")

    if st.button("Submit"):
        if query:
            # Build a RetrievalQA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=False
            )
            
            # Get response
            response = qa_chain.invoke({"query": query})
            st.write("### Answer:")
            st.write(response["result"])
        else:
            st.error("Please enter a query!")

    # Clean up temporary file
    os.remove(temp_file_path)
