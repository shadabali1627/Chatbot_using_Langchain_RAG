import streamlit as st
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY environment variable not set. Please set it in the .env file.")
    st.stop()

# Initialize embeddings
try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=GOOGLE_API_KEY
    )
except Exception as e:
    st.error(f"Failed to initialize embeddings: {str(e)}")
    st.stop()

# Initialize LLM
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        google_api_key=GOOGLE_API_KEY,
    )
    direct_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        google_api_key=GOOGLE_API_KEY,
    )
except Exception as e:
    st.error(f"Failed to initialize LLM: {str(e)}")
    st.stop()

# Define the prompt template
prompt_template = """
You are a knowledgeable assistant answering questions based on the provided context from a general knowledge dataset.
If the context doesn't contain the answer, clearly state: "I couldn't find the answer in the dataset, but here's what I know from Gemini LLM:" 
and provide a helpful answer using the Gemini LLM.

Context: {context}

Question: {question}

Answer:
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Function to initialize the RAG pipeline
@st.cache_resource
def initialize_rag(_uuid=str(uuid.uuid4())):
    try:
        # Load dataset
        try:
            df = pd.read_csv("general_knowledge_qa.csv")
        except FileNotFoundError:
            st.error("Dataset file 'general_knowledge_qa.csv' not found in the current directory.")
            return None
        
        # Check for required columns
        required_columns = ['question', 'answer']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in dataset: {missing_columns}")
        
        # Clean dataset: remove rows with missing question or answer
        df = df.dropna(subset=required_columns)
        if df.empty:
            st.error("Dataset is empty or contains no valid question-answer pairs.")
            return None
        
        # Convert dataset to LangChain documents
        documents = [
            Document(page_content=f"Question: {row['question']}\nAnswer: {row['answer']}")
            for _, row in df.iterrows()
        ]
        
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        # Create vector store with FAISS
        vector_store = FAISS.from_documents(texts, embeddings)
        
        # Initialize the RAG chain
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return rag_chain
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {str(e)}")
        return None

def get_gemini_response(question):
    """Get response directly from Gemini LLM when RAG doesn't know"""
    try:
        response = direct_llm.invoke(question)
        return response.content
    except Exception as e:
        st.warning("Failed to fetch additional information from Gemini LLM.")
        return f"Error querying Gemini LLM: {str(e)}"

# Main app
def main():
    st.title("General Knowledge Q&A Assistant")
    
    # Initialize RAG pipeline
    rag_chain = initialize_rag()
    
    if rag_chain is None:
        st.error("Q&A system could not be initialized. Please check the dataset and configurations.")
        return

    # Sidebar with example questions
    st.sidebar.header("Example Questions")
    example_questions = [
        "What is the largest ocean in the world?",
        "Who invented the telephone?",
        "How many planets are there in our solar system?",
        "Which animal is known as the Ship of the Desert?"
    ]
    
    for question in example_questions:
        if st.sidebar.button(question, key=f"example_{question}"):
            st.session_state['user_input'] = question
    
    # User input
    user_input = st.text_input("Ask a question:", value=st.session_state.get('user_input', ''), key="user_input")
    
    if st.button("Get Answer"):
        if not user_input.strip():
            st.warning("Please enter a valid question.")
            return
        
        with st.spinner("Fetching answer..."):
            try:
                # Query the RAG pipeline using the updated invoke method
                result = rag_chain.invoke({"query": user_input})
                answer = result["result"]
                
                # Check if answer indicates uncertainty
                uncertain_phrases = ["i don't know", "i couldn't find", "not in my knowledge"]
                if any(phrase in answer.lower() for phrase in uncertain_phrases):
                    gemini_answer = get_gemini_response(user_input)
                    answer = f"{answer}\n\nAdditional information from Gemini LLM:\n{gemini_answer}"
                
                # Display the answer
                st.subheader("Answer:")
                st.write(answer)
                
                # Display source documents if available
                if "source_documents" in result and result["source_documents"]:
                    with st.expander("Source Information"):
                        st.write("The answer was generated based on the following dataset entries:")
                        for doc in result["source_documents"]:
                            st.write(doc.page_content)
                            st.write("---")
                else:
                    st.info("No specific dataset entries were used for this response.")
                    
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")

if __name__ == "__main__":
    main()