**Truist Chatbot - Local Deployment**<br />
This folder contains the code and configuration for The Truist Demo Agent Executed on a Langchain Framework, Database created using LlamaIndex, And Qdrant Vector Store kept locally in a docker container. 
It incorporated the retrieval and reranking techniques of Akshara's Llama Index App and hybrid Semantic Search And Sparse text search strategies.
<br />
**Folder Structure:**<br />
*Retrieval_Reranking.py*<br />
Main Python module implementing search, retrieval, and reranking logic using Semantic Comparison and Sparse Text Searches. 
It connects to the Qdrant vector database and supports a Sentence Transformers ReRanking Model. The Agent chooses between conducting a Semantic Search (Via Vector Embeddings) or A Sparse Text (Word)
search of the document based on the user input. I also set the search strategy of the agent by setting it to Hybrid Search in the State Configuration. 
This allows it to perform the dual search.
<br />
**docker-compose.yml**<br />
Docker Compose file for running a local Qdrant instance. Maps storage to the local filesystem and exposes HTTP/gRPC APIs. Docker app downloaded on docker.com and installed for Windows Powershell
<br />
**qdrant_storage/**<br />
Directory for persistent Qdrant data. Contains subfolders for collections and configuration files.
<br />
**collections/finance_documents/config.json**<br />
Configuration for the finance_documents collection, specifying vector types,chunk size, vector lenght, storage, and optimization settings. 
The reason for the docker container is to create a local Qdrant dashboard to create the collection that connects the Generation.py 
function to extract the text retrieve and save them into the vector store
<br />
**raft_state.json**<br />
Raft consensus state for Qdrant, used for cluster management (single-node in this setup). 
It is an algorithm that ensures that the JSON (Text) nodes created during the text splitting and chunking of the document remain accurate and ordered in case of failure.
It also ensures that any changes made to the Text Database are saved and acknolewdge by the other vector nodes when saving their state. In case of failure during retrieval of nodes, the algortihm reorganizes the text node store to isoloate the failed node and kieep the retrieval oepration running 
<br />
**Main Features**<br />
**Semantic and Hybrid Search:**<br />
Retrieve relevant documents using dense (semantic) or combined dense+sparse (hybrid) embeddings. The state has been preconfigured by me to run dual searches, like I said 
<br />
**Reranking:**<br />
Supports multiple reranking models (e.g., cross-encoder, BGE) to improve result quality. 
However, since the original app did not come with a specified model to default to, it will default to the cross-ecnoder model when conducting searches
<br />
**Qdrant Integration:**<br />
Uses Qdrant as a vector database for fast and scalable similarity search.
<br />
**Environment Configuration:**<br />
Uses .env file for API keys, URLs, and collection names. A unique LLM key or Ollama instance variable will need to be coded here.
<br />
**agentic_chatbot_qdrant**<br />
**!!!!!!Core file to interact with langchain function performing dual search of Qdrant vector store !!!!!!**
<br />
**Quick Start**<br />
Set up environment variables:<br />
Create a .env file with the following keys:<br />
<details><summary>Click to expand</summary>
Qdrant_API_KEY=your-api-key<br />
Qdrant_URL=http://localhost<br />
Collection_Name=finance_documents<br />
Qdrant_port=6333<br />
Qdrant_grpc=6334<br />
</details>
<br />
Start Qdrant Container Locally:<br />
<details><summary>Click to expand</summary>
In Terminal: docker-compose up -d
</details>
<br />
Install Python dependencies:<br />
<details><summary>Click to expand</summary>
In terminal: pip install -r requirements.txt
</details>
<br />
Run the retrieval script:<br />
<details><summary>Click to expand</summary>
In Terminal: python Retrieval_Reranking.py
</details>
<br />
Requirements<br />
Python 3.12+<br />
Docker & Docker Compose<br />
See requirements.txt for Python dependencies. <br />
Open local terminal or right click on File Directory, open in integrated terminal, pip install requirement.txt<br />
Notes<br />
The system is designed for local development and testing.<br />
For production, review security and scaling considerations.<br />
Qdrant data is persisted in the qdrant_storage directory.<br />
References<br />
https://qdrant.tech/documentation/quickstart/<br />
https://sbert.net/<br />
https://sbert.net/examples/cross_encoder/applications/README.html<br />
https://www.llamaindex.ai/blog/using-llms-for-retrieval-and-reranking-23cf2d3a14b6<br />
https://docs.llamaindex.ai/en/stable/examples/vector_stores/qdrant_hybrid/<br />
https://python.langchain.com/docs/how_to/agent_executor/<br />
https://python.langchain.com/api_reference/community/index.html<br />
https://python.langchain.com/api_reference/core/index.html<br />
https://python.langchain.com/docs/integrations/llms/openai/<br />
https://python.langchain.com/docs/integrations/llms/openai/<br />
LlamaIndex<br />

