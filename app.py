import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.chains import StuffDocumentsChain
from langchain.chains import LLMChain
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Default legal documents
DEFAULT_DOCS = ["indian-penal-code.pdf", "20240716890312078.pdf"]

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text 

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)        
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, is_new_files=False):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    if os.path.exists("faiss_index"):
        try:
            # Load the existing vector store and add only new file chunks
            vector_store = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
            if is_new_files and text_chunks:
                vector_store.add_texts(text_chunks)
        except Exception as e:
            st.sidebar.error(f"Error loading or updating FAISS index: {str(e)}")
    else:
        try:
            # Create a new vector store with all chunks (default + uploaded)
            vector_store = FAISS.from_texts(text_chunks, embeddings=embeddings)
        except Exception as e:
            st.sidebar.error(f"Error creating FAISS index: {str(e)}")

    # Save updated or new FAISS index
    try:
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.sidebar.error(f"Error saving FAISS index: {str(e)}")

def get_conversational_chain():
    # Context including the document details and queries
    context = """
    You are an AI model designed to assist as a legal consultant knowledgeable about the Constitution of India and the Indian Penal Code. You are provided with one or more PDF files containing legal documents of individuals, along with a legal query related to those documents. Your task is to extract relevant legal information from the provided documents and answer the legal query with accuracy and detail.
    """

    # Prompt Template: Customizable structure for each query
    prompt_template = """
    ### Inputs:
    - **PDF Legal Document(s)**: {documents}
    - **Query**: {query}

    ### Steps:
    1. **Understand the Query**: Carefully read and comprehend the legal query or issue presented by the user.
    2. **Extract Relevant Information**: Analyze the provided PDFs to identify and extract pertinent information or clauses relevant to the query.
    3. **Identify Relevant Laws**: Determine which sections of the Constitution of India or the Indian Penal Code apply to the query.
    4. **Research or Recall Details**: Use your knowledge or reference materials to gather detailed information about the relevant laws and their interpretations.
    5. **Analyze the Situation**: Consider how the laws and the information from the documents apply to the specific circumstances of the query.
    6. **Provide Clear Advice**: Offer a well-structured and legally sound response or advice, ensuring it addresses all aspects of the query.
    7. **Include Citations**: If applicable, refer to specific articles, sections, or precedents that support the response.

    ### Output Format:
    Provide a detailed and structured paragraph response that includes citations to specific legal provisions when possible.
    Ensure that the advice is legally accurate.
    """

    # Initialize the Google Generative AI model
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    # Create a prompt template that dynamically accepts documents and user queries
    prompt = PromptTemplate(template=prompt_template, input_variables=["documents", "query"])

    # Create a question-answering chain using the model
    chain = load_qa_chain(llm=model, chain_type="stuff")

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
    except Exception as e:
        st.error(f"Error retrieving similar documents: {str(e)}")
        return ""

    chain = get_conversational_chain()
    
    try:
        # Pass documents and question in the expected structure
        response = chain({
            "input_documents": docs,
            "context": " ".join(doc.page_content for doc in docs),  # Joining all text from similar docs
            "question": user_question
        }, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return ""

def main():
    st.set_page_config("Solve your legal queries")
    
    # Sidebar for PDF uploads
    st.sidebar.title("Upload PDF Files")
    uploaded_files = st.sidebar.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    # Check if the FAISS index already exists
    is_faiss_index_exists = os.path.exists("faiss_index")

    if uploaded_files:
        try:
            # If there are uploaded files, process them directly from in-memory file-like objects
            pdf_text = get_pdf_text(uploaded_files)  
            text_chunks = get_text_chunks(pdf_text)
            
            # Add only new documents to the existing vector store if it exists, else create a new one
            get_vector_store(text_chunks, is_new_files=is_faiss_index_exists)
            st.sidebar.success("New PDF files uploaded and processed successfully!")
        except Exception as e:
            st.sidebar.error(f"Error processing uploaded files: {str(e)}")
    else:
        if not is_faiss_index_exists:
            try:
                # If there are no uploaded files and no existing index, process the default docs
                pdf_text = get_pdf_text(DEFAULT_DOCS)
                text_chunks = get_text_chunks(pdf_text)
                get_vector_store(text_chunks)
                st.sidebar.success("Default legal documents processed successfully!")
            except Exception as e:
                st.sidebar.error(f"Error processing default files: {str(e)}")
        else:
            st.sidebar.info("FAISS index already exists, no default documents processed again.")

    # Conversation history
    if 'history' not in st.session_state:
        st.session_state.history = []

    # User question input
    user_question = st.text_input("Enter your legal query:")
    if user_question:
        # Get response and remember the conversation
        response_text = user_input(user_question)
        st.session_state.history.append((user_question, response_text))
        
    # Display conversation history
    st.write("### Conversation History")
    for q, a in st.session_state.history:
        st.write(f"**Q:** {q}")
        st.write(f"**A:** {a}")

if __name__ == "__main__":
    main()
