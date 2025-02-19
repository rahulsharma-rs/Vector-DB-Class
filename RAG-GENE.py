import os
import streamlit as st
from dotenv import load_dotenv

# ----------------------------------------------------
# Step 1: Load Environment Variables
# ----------------------------------------------------
load_dotenv()
# Ensure your OpenAI API key is set in your environment
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()

st.title("Gene Metadata Chatbot with RAG (OpenAI & LangChain)")

# ----------------------------------------------------
# Step 2: Load Gene Metadata CSV and Prepare Data
# ----------------------------------------------------
import pandas as pd

csv_filename = "cancer_gene_data_100.csv"
try:
    df = pd.read_csv(csv_filename)
except Exception as e:
    st.error(f"Error loading CSV file: {e}")
    st.stop()

# Ensure the CSV has the required columns.
required_cols = ["Gene Name", "Associated Cancer", "Pathway Involved", "Reference"]
if not all(col in df.columns for col in required_cols):
    st.error(f"CSV must contain the following columns: {required_cols}")
    st.stop()

# Create a combined description for each gene.
def create_description(row):
    return (
        f"Gene: {row['Gene Name']}. "
        f"Associated Cancer: {row['Associated Cancer']}. "
        f"Pathway: {row['Pathway Involved']}. "
        f"Reference: {row['Reference']}."
    )

df["Description"] = df.apply(create_description, axis=1)
st.write("### Sample Gene Descriptions")
st.write(df[["Gene Name", "Description"]].head())

# Sidebar: Show gene metadata for reference.
st.sidebar.header("Gene Metadata")
for idx, row in df.iterrows():
    st.sidebar.markdown(f"**{row['Gene Name']}**: {row['Associated Cancer']} | {row['Pathway Involved']}")

# ----------------------------------------------------
# Step 3: Build the Vector Store with LangChain
# ----------------------------------------------------
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Convert each row into a Document object.
docs = [
    Document(page_content=row["Description"], metadata={"Gene Name": row["Gene Name"]})
    for _, row in df.iterrows()
]

# Initialize the Hugging Face embedding model.
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a FAISS vector store from the documents.
vector_store = FAISS.from_documents(docs, embedding_model)

st.write("Vector store created successfully.")

# ----------------------------------------------------
# Step 4: Initialize the Generative Model using OpenAI's Chat Model
# ----------------------------------------------------
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=openai_api_key,
    temperature=0.7
)

# ----------------------------------------------------
# Step 5: Create the RAG Chain
# ----------------------------------------------------
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff" concatenates retrieved documents to the prompt.
    retriever=vector_store.as_retriever()
)

st.write("RetrievalQA chain created successfully.")

# ----------------------------------------------------
# Step 6: Build the Chatbot Interface
# ----------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.subheader("Chat with the Gene Metadata Bot")
user_query = st.text_input("Ask a question about the gene data:")

if user_query:
    with st.spinner("Generating answer..."):
        answer = qa_chain.run(user_query)
    st.session_state.chat_history.append({"user": user_query, "bot": answer})

if st.session_state.chat_history:
    st.markdown("### Conversation History")
    for chat in st.session_state.chat_history:
        st.markdown(f"**User:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")
        st.markdown("---")
