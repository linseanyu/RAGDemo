import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import os

#### Prepare Data ####

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

# Check if FAISS index already exists
if os.path.exists("faiss_index") and os.path.isdir("faiss_index"):
    print("Loading existing FAISS index...")
    vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
else:
    print("Creating new FAISS index...")
    # Load the data
    products_df = pd.read_csv("products.csv")
    conversations_df = pd.read_csv("conversations.csv")

    # Convert product data to text
    def format_product(row):
        return f"Product ID: {row['product_id']} | Name: {row['name']} | Description: {row['description']} | Price: ${row['price']}"

    products_text = products_df.apply(format_product, axis=1).to_list()

    # Convert conversation data to text

    def format_conversation(row):
        return f"User: {row['customer_message']}\nAssistant: {row['agent_response']}"

    conversations_text = conversations_df.apply(format_conversation, axis=1).to_list()

    # Combine all text

    all_text = products_text + conversations_text

    # Create metadata for each text
    product_metadata = [{"source": "product", "product_id": row["product_id"]} for _, row in products_df.iterrows()]
    conversation_metadata = [{"source": "conversation", "conversation_id": row["conversation_id"]} for _, row in conversations_df.iterrows()]
    all_metadata = product_metadata + conversation_metadata

    #### Chunk Data ####

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    chunked_texts = []
    chunked_metadata = []

    # Chunk the data
    for text, meta in zip(all_text, all_metadata):
        chunks = text_splitter.split_text(text)
        chunked_texts.extend(chunks)
        chunked_metadata.extend([meta] * len(chunks))

    #### Embed Data ####

    # Batch embedding function
    def batch_embed(texts, batch_size=128):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = embedding_model.embed_documents(batch)
            embeddings.extend(batch_embeddings)
        return embeddings

    # Embed all chunks
    chunk_embeddings = batch_embed(chunked_texts)

    #### Index Embeddings with FAISS ####

    # Create FAISS index
    vector_store = FAISS.from_embeddings(
        text_embeddings=list(zip(chunked_texts, chunk_embeddings)),
        embedding=embedding_model,
        metadatas=chunked_metadata
    )

    #### Persist Index ####

    vector_store.save_local("faiss_index")

#### Set Up Retrieval ####

# Create retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Return top 5 chunks
)

#### Build RAG Pipeline ####

# Initialize language model
llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-base",
    task="text2text-generation",
    pipeline_kwargs={"max_length": 512, "truncation": True}
)

# Create RAG pipeline
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Combines retrieved docs into a single prompt
    retriever=retriever,
    return_source_documents=True
)

# Interactive question answering loop
print("\n=== RAG Product Assistant ===")
print("Ask questions about products or past conversations (type 'exit' to quit)")
print("Example: 'What is the battery life of the Wireless Headphones?'\n")

while True:
    user_query = input("Your question: ")
    
    if user_query.lower() in ["exit", "quit", "q"]:
        print("Goodbye!")
        break
    
    if not user_query.strip():
        continue
        
    print("\nSearching for answer...")
    result = qa_chain.invoke({"query": user_query})
    
    print(f"\nAnswer: {result['result']}")