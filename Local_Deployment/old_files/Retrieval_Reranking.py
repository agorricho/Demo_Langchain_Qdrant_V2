from sentence_transformers import CrossEncoder
import logging
from dotenv import load_dotenv
load_dotenv()
import os
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import SparseVector
from typing import List, Union
from sentence_transformers import CrossEncoder

from typing import List
import pprint

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI 
from llama_index.agent.openai import OpenAIAgent
from pydantic import BaseModel

Qdrant_API_KEY = os.getenv('Qdrant_API_KEY')
Qdrant_URL = os.getenv('Qdrant_URL')
Collection_Name = os.getenv('Collection_Name')
Qdrant_port=int(os.getenv('Qdrant_port'))
Qdrant_grpc=int(os.getenv('Qdrant_grpc'))

# Define the reranker models
class SentenceTransformerRerank:
    def __init__(self, model, top_n):
        self.model = CrossEncoder(model)
        self.top_n = top_n

    def rerank(self, query, documents):
        # Compute the similarity scores between the query and each document
        scores = self.model.predict([(query, doc) for doc in documents])

        # Sort the documents based on their similarity scores
        ranked_documents = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        # Select the top documents
        top_documents = [doc for doc, score in ranked_documents[:self.top_n]]

        return top_documents

# Dictionary of reranker models
RERANKERS = {
    "cross-encoder": SentenceTransformerRerank(model='cross-encoder/ms-marco-MiniLM-L-6-v2', top_n=2),
    "BGE": SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=2),
    "bge-reranker-large": SentenceTransformerRerank(model="BAAI/bge-reranker-large", top_n=2)
}

# ReRankingAgent function
def ReRankingAgent(query, documents, reranking_model: str):
    # Get the reranker model based on user preference
    reranker = RERANKERS.get(reranking_model)

    if reranker is None:
        # If no reranker is specified, return the documents as is
        return documents

    # Perform reranking
    top_documents = reranker.rerank(query, documents)

    return top_documents

#  Search Strategy Interface
class SearchStrategy:
    def search(self, query: str) -> List[str]:
        raise NotImplementedError

class SemanticSearch(SearchStrategy):
    def query_semantic_search(self, query: str) -> List[str]:
        # Load the dense embedding model
        embedding_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Initialize the Qdrant client
        qdrant_client = QdrantClient(
            url=Qdrant_URL,
            api_key=Qdrant_API_KEY,
            port=Qdrant_port,
            grpc_port=Qdrant_grpc,
            timeout=30
        )

        # Embed the query using the dense embedding model
        dense_query = list(embedding_model.embed([query]))[0].tolist()

        # Perform the semantic search
        results = qdrant_client.query_points(
                collection_name=Collection_Name,
                query=dense_query,
                using="dense",
                limit=4,
            )
            
        documents = [point.payload['text'] for point in results.points]

        return documents

class HybridSearch(SearchStrategy):
    # Load environment variables
    
    
    def query_hybrid_search(self, query: str) -> List[str]:

        embedding_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        qdrant_client = QdrantClient(
            url=Qdrant_URL,
            api_key=Qdrant_API_KEY,
            port=Qdrant_port,
            grpc_port=Qdrant_grpc,
            timeout=30
        )

        # Embed the query using the dense embedding model
        dense_query = list(embedding_model.embed([query]))[0].tolist()

        # Embed the query using the sparse embedding model
        sparse_query = list(sparse_embedding_model.embed([query]))[0]

        results = qdrant_client.query_points(
            collection_name=Collection_Name,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_query.indices.tolist(), values=sparse_query.values.tolist()),
                    using="sparse",
                    limit=4,
                ),
                models.Prefetch(
                    query=dense_query,
                    using="dense",
                    limit=4,
                ),
            ],
            
            query=models.FusionQuery(fusion=models.Fusion.RRF), #Reciprocal Rerank Fusion
        )
        
        # Extract the text from the payload of each scored point
        documents = [point.payload['text'] for point in results.points]

        return documents

'''
def metadata_filter(file_names: Union[str, List[str]]) -> models.Filter:
    
    if isinstance(file_names, str):
        
        file_name_condition = models.FieldCondition(
            key="file_name",
            match=models.MatchValue(value=file_names)
        )
    else:
        
        file_name_condition = models.FieldCondition(
            key="file_name",
            match=models.MatchAny(any=file_names)
        )

    return models.Filter(
        must=[file_name_condition]
    )
'''

# Factory Function to Get the Appropriate Search Strategy
def get_search_strategy(search_type: str) -> SearchStrategy:
    if search_type == 'semantic':
        return SemanticSearch()
    elif search_type == 'hybrid':
        return HybridSearch()
    else:
        raise ValueError("Invalid search type")

class Retriever():
    def __init__(self, state: dict):
        self.state = state
        self.query = state.get('query')
        if state.get('search_type'):
            self.search_type = state.get('search_type')
        else:
            self.search_type = 'semantic'
        if state.get('reranking_model'):
            self.reranking_model = state.get('reranking_model')
        else:
            self.reranking_model = 'cross-encoder'

    def retriever(self):
        """
        Perform the search and retrieval process based on the specified search type, query, and reranking model.
        """
        print("Starting the search and retrieval process")
        search_strategy = get_search_strategy(self.search_type)
        if self.search_type == 'semantic':
            documents = search_strategy.query_semantic_search(self.query)
        elif self.search_type == 'hybrid':
            documents = search_strategy.query_hybrid_search(self.query)
        else:
            raise ValueError("Invalid search type")
        print("Search and retrieval process completed")
        reranked_documents = ReRankingAgent(self.query, documents, self.reranking_model)
        print("Reranking of the retrieved documents is complete")

        return reranked_documents
    
if __name__=='__main__':
    state={'query':'What is the LifeForce program?'}
    doc_ret=Retriever(state)
    reranked_docs=doc_ret.retriever()
    print(f'Number of reranked documents: {len(reranked_docs)}')
