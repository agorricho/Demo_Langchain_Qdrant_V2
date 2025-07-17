import os
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter, SparseVector
from fastembed import TextEmbedding, SparseTextEmbedding

# Load environment variables
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
OLLAMA_URL = os.getenv("OLLAMA_URL")

# Initialize dense and sparse embedding models
dense_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# Initialize LLM Ollama
llm_ollama = ChatOllama(
    model="dolphin3:latest",  # options: llama3.1:latest, dolphin3.1:latest, llava-llama3.1:latest
    base_url=OLLAMA_URL,
    temperature=0
)

# Prompt template for the agent
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

# Hybrid retrieval tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Perform hybrid (dense + sparse) search using Qdrant Query API."""
    dense_emb = list(dense_model.embed([query]))[0]
    sparse_emb = list(sparse_model.embed([query]))[0]
    sparse_vector = models.SparseVector(
        indices=sparse_emb.indices.tolist(),
        values=sparse_emb.values.tolist()
    )
    context = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=sparse_vector,
                using="sparse",
                limit=20,
            ),
            models.Prefetch(
                query=dense_emb,
                using="dense",
                limit=20,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
        limit=5
    )
    serialized = "\n\n".join(
        (f"ID: {pt.id}\nScore: {pt.score}\nText: {pt.payload['text']}\nSource: {pt.payload['file_name']}\nPage: {pt.payload['page_number']}")
        for pt in context.points
    )
    return serialized, context

def main():
    tools = [retrieve]
    agent = create_react_agent(llm=llm_ollama, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    print("Welcome to the Truist Hybrid Chatbot (Ollama + Qdrant)")
    print("Type your question and press Enter. Type 'exit' to quit.")
    while True:
        user_input = input("\nYour question: ")
        if user_input.strip().lower() in ("exit", "quit"): break
        response = agent_executor.invoke({"input": user_input})
        print("\nAnswer:")
        print(response.get('output', response))

if __name__ == "__main__":
    main() 