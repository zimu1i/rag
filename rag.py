"""
Muzi Li
2026.06.25
Data Science Student at UWaterloo
"""
import pymupdf
import numpy as np
import json
import os
import nltk
from openai import OpenAI
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize


# perform RAG on "Canada Business Corporations Act (R.S.C., 1985, c. C-44)"
doc = pymupdf.open("C-44.pdf")

full_text = ""
for page in doc:
    full_text += page.get_text()

# chunking
def chunk_text(text, chunk_size=500, overlap_sent=1):
    sentences = sent_tokenize(text)

    chunks=[]
    curr_chunk=[]
    curr_len=0
    
    for sentence in sentences:
        sentence_len = len(sentence)

        if curr_len + sentence_len > chunk_size and curr_chunk:
            chunks.append(" ".join(curr_chunk))
            curr_chunk = curr_chunk[-overlap_sent:]
            curr_len = sum(len(s) for s in curr_chunk)

        curr_chunk.append(sentence)
        curr_len += sentence_len
    
    if curr_chunk:
        chunks.append(" ".join(curr_chunk))
    
    return chunks
    
chunks = chunk_text(full_text)

# print total chunks
# print(f'Total {len(chunks)} chunks')
# print(chunks[0])

# openai embedding model
client = OpenAI()

def get_embedding(text):
    response = client.embeddings.create(
        model='text-embedding-3-small',
        input=text
    )
    return response.data[0].embedding

# store embeddings
def store_embeddings(chunks, embeddings, filename="embeddings.json"):
    data = {"chunks":chunks,
            "embeddings":embeddings}
    with open(filename, "w") as f:
        json.dump(data, f)
    print(f"Saved to {filename}")

# load embeddings 
def load_embeddings(filename="embeddings.json"):
    with open(filename, "r") as f:
        data = json.load(f)
    return data["chunks"], data["embeddings"]

# checks if embeddings have already been generated
if os.path.exists("embeddings.json"):
    print("Loading saved embeddings...")
    chunks, embeddings = load_embeddings()
else:
    print("Generating embeddings...")
    chunks = chunk_text(full_text)
    embeddings = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
        if i % 10 == 0:
            print(f"Progress: {round(i/len(chunks)*100, 2)}%")
    store_embeddings(chunks, embeddings)

# calculate cosine similarity
def cos_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))

def find_chunks(question, chunks, embeddings, top_k=3):
    question_embedding = get_embedding(question)

    # calculate cosine similaritiy score for each chunk
    similarities = []
    for i, emb in enumerate(embeddings):
        score = cos_similarity(question_embedding, emb)
        similarities.append((score, i))
    
    # sort similarities to have the highest score at top
    similarities.sort(reverse=True)

    # find top k chunks in the document 
    # to input to the LLM for generating responses
    top_chunks = []
    for score, idx in similarities[:top_k]:
        top_chunks.append((score, chunks[idx]))

    return top_chunks

# use LLM to answer user inputted questions
def answer_question(question, chunks, embeddings):
    relevant = find_chunks(question, chunks, embeddings)

    print("--- Retrieved chunks ---")
    for score, chunk in relevant:
        print(f"Score: {score:.3f}")
        print(chunk[:300])
        print("---")

    context_parts = []

    for i, (score, chunk) in enumerate(relevant):
        context_parts.append(f"[Chunk {i+1}, relevance: {score:.2f}]:\n{chunk}")

    context = "\n\n".join(context_parts)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant. Answer the question based on the provided context chunks.
Each chunk has a relevance score. Prioritize higher relevance chunks.
If chunks contain reference lists or citations rather than main content, note this in your answer.
If the answer is not clearly in the context, say so."""
            },
            {
                "role": "user", 
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )

    return response.choices[0].message.content

print("\n=== RAG System Ready ===")
print("Ask questions about the paper. Type 'quit' to exit.\n")

while True: 
    question = input("Your question:")
    if question.lower() == 'quit':
        break
    answer = answer_question(question, chunks, embeddings)
    print(f"\nAnswer: {answer}\n")


