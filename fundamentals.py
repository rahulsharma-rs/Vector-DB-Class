import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# ====================================================
# Fundamentals of Vector Databases
# ====================================================

# ====================================================
# 1. Embeddings:
#    Definition and purpose:
#    Transform raw data (words) into numerical vectors.
# ====================================================

# Load spaCy's medium English model which provides 300-d word vectors.
nlp = spacy.load("en_core_web_md")

# Define a sample set of words.
words = ["apple", "banana", "cat", "dog", "computer", "python", "data", "science", "machine", "learning"]

# Obtain embeddings for these words.
embeddings = np.array([nlp(word).vector for word in words]).astype("float32")
print("=== Embeddings for Sample Words ===")
for word, vec in zip(words, embeddings):
    # Display the first five dimensions for brevity.
    print(f"{word}: {vec[:5]} ...")
print()

# ====================================================
# 2. Similarity Search:
#    - Key metrics: cosine similarity, Euclidean (L2) distance.
#    - How similarity measures help in finding “close” vectors.
# ====================================================

# Select two words to compare.
word1 = "apple"
word2 = "banana"

vec1 = nlp(word1).vector.reshape(1, -1)
vec2 = nlp(word2).vector.reshape(1, -1)

# Compute cosine similarity.
cos_sim = cosine_similarity(vec1, vec2)[0][0]

# Compute Euclidean (L2) distance.
euclidean = np.linalg.norm(vec1 - vec2)

print("=== Similarity Search ===")
print(f"Similarity between '{word1}' and '{word2}':")
print(f"Cosine Similarity: {cos_sim:.4f}")
print(f"Euclidean Distance: {euclidean:.4f}\n")

# ====================================================
# 3. Indexing Techniques:
#    Overview of indexing methods (FAISS, Annoy, HNSW).
#    Benefits of approximate nearest neighbor (ANN) searches.
# ====================================================

# We use FAISS for this demonstration.
# Get the dimensionality of our embeddings.
d = embeddings.shape[1]  # should be 300 for spaCy's en_core_web_md

# Create a FAISS index using L2 (Euclidean) distance.
index = faiss.IndexFlatL2(d)
index.add(embeddings)
print("=== FAISS Index ===")
print("FAISS index created with sample embeddings.")

# Perform a query: find the top 3 nearest neighbors to "computer".
query_word = "computer"
query_vec = nlp(query_word).vector.reshape(1, -1)
k = 3  # number of nearest neighbors

distances, indices = index.search(query_vec, k)
print(f"\nNearest neighbors for '{query_word}':")
for rank, idx in enumerate(indices[0]):
    print(f"{rank + 1}. {words[idx]} (Distance: {distances[0][rank]:.4f})")
print()

# ====================================================
# 4. Popular Tools & Comparison:
#    - Popular Tools: FAISS, Pinecone, Milvus, Qdrant.
#    - Differences between vector databases and traditional relational databases.
# ====================================================
#
# Popular Tools:
#   • FAISS: An efficient similarity search library developed by Facebook.
#   • Pinecone: A fully managed vector database service.
#   • Milvus: An open-source vector database optimized for scalable similarity search.
#   • Qdrant: A high-performance vector database designed for integration and ease-of-use.
#
# Differences:
#   • Vector databases are specialized to store and search high-dimensional vectors,
#     making them ideal for similarity search in unstructured data (e.g., images, text).
#   • Traditional relational databases excel in structured, tabular data with exact match queries.
#
# These comments serve as a conceptual overview. You can expand upon these points during your lecture.
