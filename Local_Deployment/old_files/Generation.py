from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core import PromptTemplate
from llama_index.core import Settings
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from dotenv import load_dotenv
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.tools import FunctionTool
import os
import pprint
from Retrieval_Rerankig import Retriever

load_dotenv()

def prompt_template():
    """
    Define the prompt template for generating explanations based on the context and query.
    """
    prompt_str = """
    You are a financial assistant specializing in answering questions about policies and benefits from financial documents. 
    Your task is to provide a clear, accurate, detailed, and informative explanation based on the following context and query.

    Context:
    {context_str}

    Query: {query_str}

    Instructions:
    - Summarize the answer in clear, plain language.
    - Highlight essential details such as any eligibility criteria, coverage limits, or exclusions if present.
    - Reference the specific document section(s) used.
    - If the retrieved context is insufficient, reply: "The provided documents do not contain enough information to answer this question."

    Your explanation should be informative yet accessible, suitable for someone in the Financial domain.
    Use only the provided context from retrieved documents to answer.
    If the question can be partially answered from the context, provide the answer and state the portion of the question which is not provided in the documents.
    If the answer is not found in the context, state that clearly. Cite the relevant document sections in your response.
    Response:
    """
    prompt_tmpl = PromptTemplate(prompt_str)
    return prompt_tmpl

def prompt_generation(state):
    """
    Generate the prompt for the given search type, query, and reranking model.
    """
    state = state
    retriever_agent = Retriever(state)
    reranked_documents = retriever_agent.retriever()
    query = state.get('query')
    context = "\n\n".join(reranked_documents)
    
    prompt_templ = prompt_template().format(context_str=context, query_str=query)

    return prompt_templ

class RAGStringQueryEngine(CustomQueryEngine):
    llm: OpenAI
    #response_synthesizer: BaseSynthesizer

    def custom_query(self, prompt: str) -> str:
        """
        Generate a response for the given prompt using the LLM and response synthesizer.
        """
        response = self.llm.complete(prompt)
        #summary = self.response_synthesizer.get_response(query_str=str(response), text_chunks=str(prompt))

        return str(response)
    
def create_query_engine(prompt: str):
    """
    Create a query engine for generating responses based on the given prompt.
    """
    llm = OpenAI(model="gpt-3.5-turbo")
    response_synthesizer = TreeSummarize(llm=llm)

    query_engine = RAGStringQueryEngine(
        llm=llm,
        response_synthesizer=response_synthesizer,
    )
    response = query_engine.query(prompt)
    return response.response

def generation(state):
    """
    Generate an explanation based on the given search type, query, and reranking model.
    """
    prompt = prompt_generation(state)
    print("Passing the ReRanked documents to the LLM")
    response = create_query_engine(prompt)
    print("Retrieved the response from LLMs")

    return response

def GenerationAgent(state: dict) -> OpenAIAgent:
    """
    Define the GenerationAgent for generating explanations based on the user's query, search type, and reranking model.
    """
    # def has_reranking_model(reranking_model: str) -> bool:
    #     """Useful for checking if the user has specified a reranking model."""
    #     print("checking if reranking model is specified")
    #     if reranking_model:
    #         state['reranking_model'] = reranking_model
    #     else:
    #         state['reranking_model'] = 'cross-encoder'
    #     return (state["reranking_model"] is not None)

    # def has_search_type(search_type: str) -> bool:
    #     """Useful for checking if the user has specified a search type."""
    #     print("checking if search type is specified")
    #     if search_type:
    #         state['search_type'] = search_type
    #     else:
    #         search_type='semantic'
    #     return (state["search_type"] is not None)    

    def has_query(query: str) -> bool:
        """Useful for checking if the user has specified query."""
        print("checking if query is specified")
        state['query'] = query
        return (state["query"] is not None)

    def generate_response(state) -> str:
        response = generation(state)
        print(state)
        #print(f"Response is generated and Here is the answer to your query:{response}")
        return response

    def done():
        """
        Signal that the retrieval process is complete, update the state, and return the response to the user.
        """
        state["current_speaker"] = None
        state["just_finished"] = True

    tools = [
        FunctionTool.from_defaults(fn=has_query),
        # FunctionTool.from_defaults(fn=has_search_type),
        # FunctionTool.from_defaults(fn=has_reranking_model),
        FunctionTool.from_defaults(fn=generate_response, return_direct=True),
        FunctionTool.from_defaults(fn=done),
    ]

    system_prompt = f"""
        You are a helpful assistant that is performing retrieval and generation tasks for a retrieval-augmented generation (RAG) system.
        Your task is to retrieve documents based on the user's query, search type, and reranking model, and then generate a response based on the retrieved documents.
        If the user supplies the necessary information, and make sure that has_query is not none,
        then call the tool "generate_response" using the provided details to perform the retrieval and generation process because it has the generation function.
        The current user state is:
        {pprint.pformat(state, indent=4)}
        When you have completed the generation process, call the tool "done" to signal that you are done.
        If the user asks to do anything other than retrieve documents, call the tool "done" with an empty string as an argument to signal that some other agent should help.
        """


    return OpenAIAgent.from_tools(
        tools = tools,
        llm=OpenAI(model="gpt-3.5-turbo", verify_ssl=False),
        system_prompt=system_prompt,
    )

if __name__=='__main__':
    state = {   'chunk_overlap': None,
    'chunk_size': None,
    'current_speaker': None,
    'embedding_model': None,
    'input_dir': None,
    'just_finished': False,
    'query': None,
    'reranking_model': None,
    'search_type': None,
    }
    agent = GenerationAgent(state=state)
    response = agent.chat("What are the benefits of Health Care Savings Account?")
    print(response)
    response_1=agent.chat("What are the options under mutual funds, and can you explain how to choose the right one?")
    print(response_1)
