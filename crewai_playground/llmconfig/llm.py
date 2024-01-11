from langchain.chat_models import BaseChatModel
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatAnyscale
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import VertexAI
import box
import yaml

class LLM:
    cfg: box.Box

    def __init__(self):
        with open("config/llm-config.yaml", "r", encoding="utf8") as ymlfile:
            self.cfg = box.Box(yaml.safe_load(ymlfile))
            # load_dotenv(dotenv_path=cfg.ENVDIR, verbose=True)
    
    def get_vertexai_llm(self, model_name: str) -> BaseChatModel:
        cfg = self.cfg.vertexai[model_name]
        return VertexAI(
            model_name=cfg.model_name,
            location=cfg.location,
            project=cfg.project,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            streaming=True
        )
