import os
import streamlit as st
from dotenv import load_dotenv

# ----------------------------------------------------
# Step 1: Load Environment Variables
# ----------------------------------------------------
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()

st.title("RAG Chatbot with Vector Store")

# ----------------------------------------------------
# Step 2: Define Documents and Build a Vector Store
# ----------------------------------------------------
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Define a small set of documents.
documents = {
    "apple": "Apple is a fruit that is red or green in color and is known for its crisp texture and sweet taste.",
    "banana": "Banana is a long, yellow fruit that is rich in potassium and has a soft, sweet interior.",
    "cat": "A cat is a small domesticated animal known for its agility and independent nature.",
    "dog": "A dog is a loyal pet, often considered man's best friend, known for its friendly behavior.",
    "computer": "A computer is an electronic device that processes data and performs a wide range of tasks.",
    "python": "Python is a high-level programming language known for its readability and is widely used in data science, web development, and automation.",
    "data": "Data represents information, often stored and processed in digital form, and is fundamental to analytics and decision making.",
    "science": "Science is the systematic study of the natural world through observation and experimentation.",
    "machine": "A machine refers to a tool or device that performs a specific task, especially those powered by electricity.",
    "learning": "Learning is the process of acquiring knowledge or skills through study, experience, or teaching."
}

# Convert documents into LangChain Document objects.
docs = [Document(page_content=text, metadata={"title": key}) for key, text in documents.items()]

# Initialize the embedding model (using a lightweight Hugging Face model).
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a FAISS vector store from the documents.
vector_store = FAISS.from_documents(docs, embedding_model)

# ----------------------------------------------------
# Step 3: Display the Vector Store in the Sidebar
# ----------------------------------------------------
st.sidebar.title("Vector Store Documents")
for doc in docs:
    st.sidebar.markdown(f"**{doc.metadata['title']}**: {doc.page_content}")

# ----------------------------------------------------
# Step 4: Initialize the Generative LLM and Create a RAG Chain
# ----------------------------------------------------
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Initialize ChatOpenAI with your API key.
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    openai_api_key=openai_api_key,
    temperature=0.7
)

# Build a RetrievalQA chain using the vector store as the retriever.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff" concatenates retrieved docs to the prompt.
    retriever=vector_store.as_retriever()
)

# ----------------------------------------------------
# Step 5: Create the Chatbot Interface
# ----------------------------------------------------
# Initialize session state for chat history if not already done.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.subheader("Chat with the RAG Bot")
user_input = st.text_input("Ask a question about the documents:")

if user_input:
    with st.spinner("Generating answer..."):
        answer = qa_chain.run(user_input)
    # Append the conversation to session state.
    st.session_state.chat_history.append({"user": user_input, "bot": answer})

# Display the chat history.
if st.session_state.chat_history:
    st.markdown("### Conversation")
    for chat in st.session_state.chat_history:
        st.markdown(f"**User:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")
        st.markdown("---")
