class Qdrant_vector_db():
```
Defines a new class named `Qdrant_vector_db`. This class is designed to handle the connection to a Qdrant vector database and index documents into it.

```python
    Qdrant_API_KEY = os.getenv('Qdrant_API_KEY')
    Qdrant_URL = os.getenv('Qdrant_URL')
    Collection_Name = os.getenv('collection_name')
```
These are **class variables** that read environment variables for the Qdrant API key, URL, and collection name. They are set when the class is first loaded.

```python
    qdrant_client = QdrantClient(
                                url=Qdrant_URL,
                                api_key=Qdrant_API_KEY)
```
Creates a Qdrant client instance using the URL and API key from the environment variables. This client is used to interact with the Qdrant server.

```python
    Embeddings = {
        "sentence-transformer": "sentence-transformers/all-MiniLM-L6-v2",
        "snowflake": "Snowflake/snowflake-arctic-embed-m",
        "BAAI": "BAAI/bge-large-en-v1.5",
    }
```
A dictionary mapping friendly embedding model names to their actual model identifiers. This allows you to select an embedding model by a simple name.

---

#### The main method:

```python
    def index_to_qdrant(self,
        embedding_model_name: str,
        nodes_json_path: str,
        collection_name: str = None,
        dense_vector_size: int = 384,
        sparse_model_name: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"
    ) -> None:
```
Defines a method to index documents into Qdrant. It takes parameters for the embedding model, the path to the nodes JSON file, the collection name, the dense vector size, and the sparse model name.

```python
        collection_name = collection_name or os.getenv('collection_name')
```
If `collection_name` is not provided, it falls back to the environment variable.

```python
        print("Loading nodes from JSON file...")
        try:
            with open(nodes_json_path, 'r') as file:
                nodes = json.load(file)
            documents = [node['text'] for node in nodes]
            metadata_list = [node['metadata'] for node in nodes]
            print(f"Loaded {len(nodes)} nodes from {nodes_json_path}")
        except Exception as e:
            print(f"Error loading nodes: {e}")
            raise
```
Loads the nodes (chunks of text and metadata) from the specified JSON file. Extracts the text and metadata into separate lists. If there’s an error, it prints and raises it.

```python
        if not Qdrant_vector_db.qdrant_client.collection_exists(collection_name):
            print(f"Creating collection '{collection_name}'...")
            Qdrant_vector_db.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    'dense': models.VectorParams(
                        size=dense_vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False),
                    )
                }
            )
```
Checks if the specified collection exists in Qdrant. If not, it creates the collection with the specified dense and sparse vector configurations.

```python
        dense_embedder = TextEmbedding(model_name=Qdrant_vector_db.Embeddings[embedding_model_name])
        sparse_embedder = SparseTextEmbedding(model_name=sparse_model_name)
```
Initializes the dense and sparse embedding models using the selected names.

```python
        points = []
        for idx, (doc, metadata) in enumerate(tqdm(zip(documents, metadata_list), 
                                                total=len(documents),
                                                desc="Indexing documents")):
            # Generate embeddings
            dense_embedding = list(dense_embedder.embed([doc]))[0]
            sparse_embedding = list(sparse_embedder.embed([doc]))[0]
            
            # Create sparse vector
            sparse_vector = models.SparseVector(
                indices=sparse_embedding.indices.tolist(),
                values=sparse_embedding.values.tolist()
            )
            
            # Create point structure
            points.append(models.PointStruct(
                id=idx,
                vector={
                    "dense": dense_embedding.tolist(),
                    "sparse": sparse_vector
                },
                payload={
                    "text": doc,
                    **metadata
                }
            ))
```
For each document and its metadata:
- Generates dense and sparse embeddings.
- Converts the sparse embedding to a Qdrant `SparseVector`.
- Creates a `PointStruct` (Qdrant’s data structure for a vector and its payload).
- Appends the point to the list.

```python
        Qdrant_vector_db.qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Successfully indexed {len(points)} documents in collection '{collection_name}'")
```
Uploads (upserts) all points to the specified Qdrant collection and prints a success message.

---

**Summary:**  
This class is a utility for:
- Connecting to Qdrant using environment variables.
- Loading pre-chunked documents from a JSON file.
- Generating dense and sparse embeddings for each chunk.
- Creating a Qdrant collection if needed.
- Uploading all document vectors and metadata to Qdrant for hybrid search.

Let me know if you want a more detailed breakdown of any section!class Qdrant_vector_db():
```
Defines a new class named `Qdrant_vector_db`. This class is designed to handle the connection to a Qdrant vector database and index documents into it.

```python
    Qdrant_API_KEY = os.getenv('Qdrant_API_KEY')
    Qdrant_URL = os.getenv('Qdrant_URL')
    Collection_Name = os.getenv('collection_name')
```
These are **class variables** that read environment variables for the Qdrant API key, URL, and collection name. They are set when the class is first loaded.

```python
    qdrant_client = QdrantClient(
                                url=Qdrant_URL,
                                api_key=Qdrant_API_KEY)
```
Creates a Qdrant client instance using the URL and API key from the environment variables. This client is used to interact with the Qdrant server.

```python
    Embeddings = {
        "sentence-transformer": "sentence-transformers/all-MiniLM-L6-v2",
        "snowflake": "Snowflake/snowflake-arctic-embed-m",
        "BAAI": "BAAI/bge-large-en-v1.5",
    }
```
A dictionary mapping friendly embedding model names to their actual model identifiers. This allows you to select an embedding model by a simple name.

---

#### The main method:

```python
    def index_to_qdrant(self,
        embedding_model_name: str,
        nodes_json_path: str,
        collection_name: str = None,
        dense_vector_size: int = 384,
        sparse_model_name: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"
    ) -> None:
```
Defines a method to index documents into Qdrant. It takes parameters for the embedding model, the path to the nodes JSON file, the collection name, the dense vector size, and the sparse model name.

```python
        collection_name = collection_name or os.getenv('collection_name')
```
If `collection_name` is not provided, it falls back to the environment variable.

```python
        print("Loading nodes from JSON file...")
        try:
            with open(nodes_json_path, 'r') as file:
                nodes = json.load(file)
            documents = [node['text'] for node in nodes]
            metadata_list = [node['metadata'] for node in nodes]
            print(f"Loaded {len(nodes)} nodes from {nodes_json_path}")
        except Exception as e:
            print(f"Error loading nodes: {e}")
            raise
```
Loads the nodes (chunks of text and metadata) from the specified JSON file. Extracts the text and metadata into separate lists. If there’s an error, it prints and raises it.

```python
        if not Qdrant_vector_db.qdrant_client.collection_exists(collection_name):
            print(f"Creating collection '{collection_name}'...")
            Qdrant_vector_db.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    'dense': models.VectorParams(
                        size=dense_vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False),
                    )
                }
            )
```
Checks if the specified collection exists in Qdrant. If not, it creates the collection with the specified dense and sparse vector configurations.

```python
        dense_embedder = TextEmbedding(model_name=Qdrant_vector_db.Embeddings[embedding_model_name])
        sparse_embedder = SparseTextEmbedding(model_name=sparse_model_name)
```
Initializes the dense and sparse embedding models using the selected names.

```python
        points = []
        for idx, (doc, metadata) in enumerate(tqdm(zip(documents, metadata_list), 
                                                total=len(documents),
                                                desc="Indexing documents")):
            # Generate embeddings
            dense_embedding = list(dense_embedder.embed([doc]))[0]
            sparse_embedding = list(sparse_embedder.embed([doc]))[0]
            
            # Create sparse vector
            sparse_vector = models.SparseVector(
                indices=sparse_embedding.indices.tolist(),
                values=sparse_embedding.values.tolist()
            )
            
            # Create point structure
            points.append(models.PointStruct(
                id=idx,
                vector={
                    "dense": dense_embedding.tolist(),
                    "sparse": sparse_vector
                },
                payload={
                    "text": doc,
                    **metadata
                }
            ))
```
For each document and its metadata:
- Generates dense and sparse embeddings.
- Converts the sparse embedding to a Qdrant `SparseVector`.
- Creates a `PointStruct` (Qdrant’s data structure for a vector and its payload).
- Appends the point to the list.

```python
        Qdrant_vector_db.qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Successfully indexed {len(points)} documents in collection '{collection_name}'")
```
Uploads (upserts) all points to the specified Qdrant collection and prints a success message.

---

**Summary:**  
This class is a utility for:
- Connecting to Qdrant using environment variables.
- Loading pre-chunked documents from a JSON file.
- Generating dense and sparse embeddings for each chunk.
- Creating a Qdrant collection if needed.
- Uploading all document vectors and metadata to Qdrant for hybrid search.

Let me know if you want a