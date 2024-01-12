from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatAnyscale
from langchain_mistralai.chat_models import ChatMistralAI
# from langchain_google_genai import GoogleGenerativeAI
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
import os

class LLMFactory:
    def __init__(self):
        load_dotenv()

    def get_ollama_llm(self, model: str, base_url="http://127.0.0.1:11434"):
        """Ollama models: mistral-agent, nous-hermes-2-solar, solar-agent, deepseek-agent, openhermes:7b-mistral-v2.5-q5_K_M"""
        return Ollama(
            model=model,
            base_url=base_url,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )
    
    def get_anyscale_llm(self, model: str):
        """Anyscale models: mistralai/Mixtral-8x7B-Instruct-v0.1, mistralai/Mistral-7B-Instruct-v0.1"""
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