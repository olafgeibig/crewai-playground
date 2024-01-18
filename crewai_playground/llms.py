from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatAnyscale
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAI
# from langchain_google_genai import GoogleGenerativeAI
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
import os

class LLMFactory:
    def __init__(self):
        load_dotenv()

    def get_openai_llm(self, model: str, base_url="http://api.openai.com/v1", api_key=os.getenv('OPEN_API_KEY')):
        """OpenAi models: 
        - gpt-4-1106-preview"""
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            streaming=True
        )  

    def get_ollama_llm(self, model: str, base_url="http://127.0.0.1:11434"):
        """Ollama models: openhermes, mistral, solar, codellama"""
        return Ollama(
            model=model,
            base_url=base_url,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )
    
    def get_anyscale_llm(self, model: str):
        """Anyscale models: 
        - mistralai/Mixtral-8x7B-Instruct-v0.1
        - mistralai/Mistral-7B-Instruct-v0.1"""
        return ChatAnyscale(
            model=model,
            api_key=os.getenv('ANYSCALE_API_KEY'),
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            streaming=True
        )  

    # install langchain-mistralai      
    def get_mistralai_llm(self, model: str):
        """Mistral AI models: mistral-tiny, mistral-small, mistral-medium"""
        return ChatMistralAI(
            model=model,
            mistral_api_key=os.getenv('MISTRALAI_API_KEY'),
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            streaming=True
        )
 
    def get_together_ai_llm(self, model: str):
        """Together AI models: mistralai/Mistral-7B-Instruct-v0.2, NousResearch/Nous-Hermes-2-Yi-34B """
        return ChatOpenAI(
            model=model, 
            api_key=os.getenv('TOGETHERAI_API_KEY'),
            base_url="https://api.together.xyz",
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            streaming=True
        )

    def get_azure_llm(self, model: str, deployment_id: str):
        """Azure OpenAI models: gpt-4-1106-Preview"""
        return AzureChatOpenAI(
            openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment = deployment_id,
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key = os.getenv("AZURE_OPENAI_API_KEY"),
        )

    # install langchain-google-genai
    # def get_google_ai_llm(self, model: str):
    #     """Goolgle AI models: gemini-pro"""
    #     return GoogleGenerativeAI(
    #         model=model, 
    #         google_api_key=os.getenv('GOOGLE_AI_API_KEY'),
    #     )

# from langchain_community.llms import VertexAI
# llm = VertexAI(
#     model_name="gemini-pro",
#     location="us-central1",
#     project="mygpt-383514",
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
#     streaming=True
# )