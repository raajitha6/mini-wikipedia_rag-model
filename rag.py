import ollama
import pickle
import os
from datasets import load_dataset

# Models
EMBEDDING_MODEL='hf.co/CompendiumLabs/bge-base-en-v1.5-gguf' #BGE Base English v1.5
LANGUAGE_MODEL='hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF' #Llama 3.2 1B Instruct

# Cache file for the vector database
CACHE_FILE="vector_db.pkl"

# Load dataset
ds=load_dataset("rag-datasets/rag-mini-wikipedia","question-answer")
ds=ds["test"]

# Simple in-memory vector database
VECTOR_DB=[]

def add_chunk_to_database(chunk,idx):
    text=f"Question: {chunk['question']}\nAnswer: {chunk['answer']}"

    embedding=ollama.embed(
        model=EMBEDDING_MODEL,
        input=text
    )['embeddings'][0]

    VECTOR_DB.append((idx,text,embedding))

def build_database():
    global VECTOR_DB

    if os.path.exists(CACHE_FILE):
        print(f"Loading cached vector DB from {CACHE_FILE}...")
        with open(CACHE_FILE, "rb") as f:
            VECTOR_DB = pickle.load(f)
        print(f"Loaded {len(VECTOR_DB)} chunks from cache.")
        return

    print("No cache found — building vector DB from scratch...")
    for i, chunk in enumerate(ds):
        add_chunk_to_database(chunk, i)
        print(f"Added chunk {i+1}/{len(ds)} to the database")

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(VECTOR_DB, f)
    print(f"Saved vector DB to {CACHE_FILE}.")

build_database()

def cosine_similarity(a,b):
    dot_product=sum(x*y for x,y in zip(a,b))
    norm_a=sum(x*x for x in a)**0.5
    norm_b=sum(x*x for x in b)**0.5
    return dot_product/(norm_a*norm_b)

def retrieve(query,top_n=3):
    query_embedding=ollama.embed(
        model=EMBEDDING_MODEL,
        input=query
    )['embeddings'][0]

    similarities=[]

    for idx,chunk,embedding in VECTOR_DB:
        similarity=cosine_similarity(query_embedding,embedding)
        similarities.append((idx,chunk,similarity))

    similarities.sort(key=lambda x:x[2],reverse=True)

    return similarities[:top_n]

# Chatbot loop
if __name__=="__main__":

    conversation_history = []
    MAX_HISTORY = 6

    while True:
        input_query=input("\nAsk me a question (or type 'exit'): ")

        if input_query.lower()=="exit":
            break

        retrieved_knowledge=retrieve(input_query)

        print("\nRetrieved knowledge:")
        for idx,chunk,similarity in retrieved_knowledge:
            print(f"\nSimilarity: {similarity:.4f}")
            print(chunk)

        context="\n".join(
            [chunk for idx,chunk,similarity in retrieved_knowledge]
        )

        instruction_prompt=f"""
        You are a helpful chatbot.

        Answer the user's question using ONLY the information provided in the context below.

        If the answer is not present in the context, say:
        "I could not find the answer in the provided context."

        Context:
        {context}
        """

        messages = (
            [{"role": "system", "content": instruction_prompt}]
            + conversation_history[-MAX_HISTORY:]
            + [{"role": "user", "content": input_query}]
        )

        stream=ollama.chat(
            model=LANGUAGE_MODEL,
            messages=messages,
            stream=True
        )

        print("\nChatbot response:\n")
        full_response = ""
        for chunk in stream:
            token = chunk['message']['content']
            print(token, end='', flush=True)
            full_response += token
        print("\n")

        conversation_history.append({"role": "user", "content": input_query})
        conversation_history.append({"role": "assistant", "content": full_response})

        # Trim the stored history too, so the list itself doesn't grow unbounded
        conversation_history = conversation_history[-MAX_HISTORY:]