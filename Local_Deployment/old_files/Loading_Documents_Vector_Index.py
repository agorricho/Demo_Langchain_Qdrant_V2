from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, SparseVector
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()
from typing import List
import pprint
import os
import json
import re

def run_document_preprocessing(input_dir, chunk_size, chunk_overlap, output_json="nodes.json"):
    print('Processing documents....')
    # Load documents
    documents = SimpleDirectoryReader(input_dir=input_dir).load_data()
    # Split documents into nodes
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)
    # Convert nodes to dicts and save as JSON
    nodes_dict = [node.to_dict() for node in nodes]
    with open(output_json, "w") as f:
        json.dump(nodes_dict, f, indent=2)
    return {
        "success": True,
        "message": f"Processed {len(nodes)} nodes. Saved to {output_json}",
        "nodes_saved": output_json,
        "num_nodes": len(nodes)
    }

class Qdrant_vector_db():
    Qdrant_API_KEY = os.getenv('Qdrant_API_KEY')
    Qdrant_URL = os.getenv('Qdrant_URL')
    Collection_Name = os.getenv('collection_name')
    qdrant_client = QdrantClient(
                                url=Qdrant_URL,
                                api_key=Qdrant_API_KEY)
            
    Embeddings = {
        "sentence-transformer": "sentence-transformers/all-MiniLM-L6-v2",
        "snowflake": "Snowflake/snowflake-arctic-embed-m",
        "BAAI": "BAAI/bge-large-en-v1.5",
    }

    def index_to_qdrant(self,
        embedding_model_name: str,
        nodes_json_path: str,
        collection_name: str = None,
        dense_vector_size: int = 384,
        sparse_model_name: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"
    ) -> None:
        """
        Index documents into Qdrant vector database with hybrid (dense + sparse) embeddings.
        
        Args:
            embedding_model_name: Name of the dense embedding model (e.g., "sentence-transformer")
            nodes_json_path: Path to the nodes JSON file
            collection_name: Name of the Qdrant collection (uses env var if None)
            dense_vector_size: Size of dense embeddings vector
            sparse_model_name: Name of sparse embedding model
        """
        # # Initialize Qdrant client
        # qdrant_client = QdrantClient(
        #     url=os.getenv('Qdrant_URL'),
        #     api_key=os.getenv('Qdrant_API_KEY')
        # )
        
        # Set collection name
        collection_name = collection_name or os.getenv('collection_name')
        
        # Load nodes
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

        # Create collection if not exists
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
        
        # Initialize embedding models
        dense_embedder = TextEmbedding(model_name=Qdrant_vector_db.Embeddings[embedding_model_name])
        # dense_embedder = TextEmbedding(model_name=embedding_model_name)
        sparse_embedder = SparseTextEmbedding(model_name=sparse_model_name)
        
        # Prepare points for upsert
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
        
        # Upsert points
        Qdrant_vector_db.qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Successfully indexed {len(points)} documents in collection '{collection_name}'")

if __name__=='__main__':
    
    
    result = run_document_preprocessing(
        input_dir="./Data",
        chunk_size=500,
        chunk_overlap=100,
        output_json="nodes.json"
    )

    print(result)
    
    qdrant_db=Qdrant_vector_db()
    
    qdrant_db.index_to_qdrant(
        embedding_model_name="sentence-transformer",
        nodes_json_path="nodes.json",
        collection_name="finance_documents",
        dense_vector_size=384
    )
