import pandas as pd
import numpy as np
import faiss
import spacy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 1. Data Preparation
# ----------------------------------------------------
# Read the CSV file containing gene metadata.
csv_filename = "cancer_gene_data_100.csv"
df = pd.read_csv(csv_filename)

# Ensure the CSV has the required columns.
required_cols = ["Gene Name", "Associated Cancer", "Pathway Involved", "Reference"]
if not all(col in df.columns for col in required_cols):
    raise ValueError(f"CSV file must contain the following columns: {required_cols}")

# Create a combined description for each gene.
def create_description(row):
    return (f"Gene: {row['Gene Name']}. "
            f"Associated Cancer: {row['Associated Cancer']}. "
            f"Pathway: {row['Pathway Involved']}. "
            f"Reference: {row['Reference']}.")

df["Description"] = df.apply(create_description, axis=1)
print("Sample Gene Descriptions:")
print(df["Description"].head())

# ----------------------------------------------------
# 2. Embedding Data & Dimensionality Reduction
# ----------------------------------------------------
# Load spaCy's model (produces 300-dimensional vectors).
nlp = spacy.load("en_core_web_md")

# Function to generate an embedding for a given text.
def embed_text(text):
    return nlp(text).vector

# Generate embeddings for each gene description.
embeddings = np.array([embed_text(desc) for desc in df["Description"]], dtype="float32")
print("\nOriginal Embeddings Shape:", embeddings.shape)  # Expected shape: (n_samples, 300)

# Use PCA to reduce the 300-d embeddings to 2 dimensions.
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)
print("2D Embeddings Shape:", embeddings_2d.shape)

# Visualize the 2D embeddings.
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', alpha=0.6, label='Gene Samples')
for i, (x, y) in enumerate(embeddings_2d):
    plt.text(x + 0.1, y + 0.1, str(i), fontsize=8)
plt.title("2D PCA Embeddings of Gene Descriptions")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.legend()
plt.show()

# ----------------------------------------------------
# 3. Indexing with FAISS
# ----------------------------------------------------
# The dimensionality is 2 after PCA.
dimension = embeddings_2d.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add the 2D embeddings to the FAISS index.
index.add(embeddings_2d.astype('float32'))
print("Number of embeddings indexed:", index.ntotal)

# ----------------------------------------------------
# 4. Query Example: Retrieve Nearest Neighbors
# ----------------------------------------------------
# Define a query text.
query_text = "Gene involved in DNA repair and cell cycle regulation in breast cancer"
# Embed the query using spaCy.
query_embedding = embed_text(query_text).reshape(1, -1)  # 300-d vector
# Use the same PCA transformation to reduce the query embedding to 2D.
query_embedding_2d = pca.transform(query_embedding).astype('float32')

# Retrieve the top 3 nearest neighbors.
k = 3
distances, indices = index.search(query_embedding_2d, k)

print("\nQuery Text:")
print(query_text)
print("\nNearest Neighbor Indices:", indices)
print("L2 Distances:", distances)

print("\nRetrieved Gene Data:")
for rank, idx in enumerate(indices[0]):
    print(f"\nRank {rank+1} - Index {idx}:")
    print(df.iloc[idx])

# ----------------------------------------------------
# 5. Visualizing the Query Result on the Graph
# ----------------------------------------------------
plt.figure(figsize=(10, 8))
# Plot all gene sample points
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', alpha=0.6, label='Gene Samples')

# Annotate each point with its index.
for i, (x, y) in enumerate(embeddings_2d):
    plt.text(x + 0.1, y + 0.1, str(i), fontsize=8)

# Plot the query point.
plt.scatter(query_embedding_2d[:, 0], query_embedding_2d[:, 1], c='green', marker='*', s=200, label='Query')

# Highlight the nearest neighbor points.
for idx in indices[0]:
    plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], c='red', s=100, label='Nearest Neighbor' if idx==indices[0][0] else "")

# Optionally, draw lines from the query to each nearest neighbor.
for idx in indices[0]:
    plt.plot([query_embedding_2d[0, 0], embeddings_2d[idx, 0]],
             [query_embedding_2d[0, 1], embeddings_2d[idx, 1]],
             'k--', linewidth=1)

plt.title("2D Embeddings with Query and Nearest Neighbors")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.legend()
plt.show()
