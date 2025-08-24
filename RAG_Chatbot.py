# RAG-Powered customer support chatbot using Streamlit

# 1. importing the libraries
import pandas as pd #data handling
import re #text preprocessing
from sentence_transformers import SentenceTransformer # embedding model
import faiss # vector search
from openai import OpenAI # openai 
import streamlit as st
from pathlib import Path

# configure streamlit page layout and title
st.set_page_config(page_title="RAG Chatbot")
st.title("RAG-Powered Customer Support Chatbot")

# 2. Load & clean dataset
# function to clean text
def clean(s):
    s = re.sub(r"\{\{.*?\}\}", "", str(s)) # remove placeholders like {{order number}}
    s = re.sub(r"\s+", " ", s).strip() # normalize spaces and remove leading/traiing whitespace
    return s.lower() # convert to lowercase

# load and cache dataset after cleaning instructions  & response
@st.cache_resource
def load_data():
    df = pd.read_csv("Customer_Support_Training_Dataset.csv")
    df["instruction_clean"] = df["instruction"].apply(clean)
    df["response_clean"] = df["response"].apply(clean)
    return df

df = load_data()

# 3. Chunk & embedd
def chunk_words(text, n=120, overlap=20):
    words  = text.split() 
    step = n - overlap
    return [" ".join(words[i:i+n]) for i in range(0, len(words), step)]

# build and cache list of mini documents
@st.cache_resource
def build_chunks(df):
    docs = []
    for rid, row in df.iterrows():
        text = f"instruction: {row['instruction_clean']} | response: {row['response_clean']}"
        for i, ch in enumerate(chunk_words(text)):
            docs.append({"rid":rid, "chunk_id": i, "text": ch})
    return docs

mini_docs = build_chunks(df)
chunk_texts = [d["text"] for d in mini_docs]

# load sentence-transformer model and compute normalized embeddings
@st.cache_resource
def get_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X = model.encode(chunk_texts, normalize_embeddings=True).astype("float32")
    return model, X

embedder, X = get_embeddings()
# create a FAISS index using inner product
index = faiss.IndexFlatIP(X.shape[1])

# Add all embeddings to index
index.add(X)

# 4. OpenAI Setup
# Loading your API key
# load API key and initialize client
api_key = Path("/Users/adityakumar/OPENAI_API_KEY.txt").read_text().strip()
client = OpenAI(api_key=api_key)

# specify model
LLM_MODEL = "gpt-4o-mini"

# 5. RAG Function
# main RAG answer function
def answer_with_rag(query, k=3, pool=30, temperature=0.2, max_tokens=300):

    # Step 1: Embed the user question
    qv = embedder.encode([query], normalize_embeddings=True).astype("float32")

    # Step 2: Retrive top 'pool' similar chunks from FAISS
    _, I = index.search(qv, pool)

    # Step 3: Pick top k chunks 
    hits = [mini_docs[int(idx)] for idx in I[0][:k]]

    # Step 4: Format selected chunks into context blocks
    context = "\n\n".join([
        f"[Doc {i+1}] (rid ={h['rid']}, chunk={h['chunk_id']}\n{h['text']}"
        for i, h in enumerate(hits)
    ])

    # Step 5: Define assistant instructions
    system_msg = (
        "You are a sarcastic customer support assistant.\n"
        "Use only the provided context to answer the user's question.\n"
        "If the answer is not available in the context, say: 'Sorry, I don't have that information.' and tell a poor joke. \n"
        "Be polite, concise, and cite sources like [Doc 1], [Doc 2] when relevant."
    )

    # Step 6: Create final prompt to send to OpenAI
    user_prompt = f"""Context:
    {context}

    Question: {query}

    Answer:"""

    # Step 7: Generate response using OpenAI
    response = client.chat.completions.create(
        model = LLM_MODEL,
        messages = [
            {"role": "system", "content":system_msg},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Step 8: Extract and return the generated answer and source chunks
    answer = response.choices[0].message.content.strip()
    return answer, hits

# 6. Streamlit Chat Interface
# enables a chat-style UI with memory of user/assistant messages

# initialize chat history in session state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# input box for user's next question
query = st.chat_input("Ask any question...")

if query:
    #store user message
    st.session_state.messages.append({"role": "user", "content": query})

    # display the user message
    with st.chat_message("user"):
        st.markdown(query)

    # LLM response box with spinner during search
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            answer, hits = answer_with_rag(query)
            st.markdown(answer)

        # expandable section to show source chunks
        with st.expander("Context Chunks Used"):
            for i, h in enumerate(hits, 1):
                st.markdown(f"**[Doc {i} rid={h["rid"]} | chunk={h['chunk_id']}**")
                st.markdown(h["text"])

    # store assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})



