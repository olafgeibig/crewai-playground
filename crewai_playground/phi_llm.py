from phi.assistant import Assistant
from phi.tools.duckduckgo import DuckDuckGo
from phi.llm.ollama import Ollama
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models.base import LanguageModelInput

# assistant = Assistant(tools=[DuckDuckGo()], show_tool_calls=True)
# assistant.print_response("Whats happening in France?")

from typing import Any, List, Optional

# Assuming necessary imports for Ollama and BaseMessage construction

class ChatPhidataAdapter(BaseChatModel):
    def __init__(self, model, base_url):
        super().__init__()
        self.model = model
        self.base_url = base_url
        # Initialize the Ollama instance here
        self.ollama = self.initialize_ollama(model, base_url)

    def initialize_ollama(self, model, base_url):
        # Assuming Ollama initialization goes here
        return Ollama(model=model, base_url=base_url)

    def invoke(self, input: LanguageModelInput, config: Optional[RunnableConfig] = None, *, stop: Optional[List[str]] = None, **kwargs: Any) -> BaseMessage:
        # Extract prompt from LanguageModelInput
        prompt = input.text  # Assuming 'text' is a property of LanguageModelInput
        
        # Invoke Ollama with the prompt and any relevant kwargs
        # Assuming Ollama's method to generate text accepts similar parameters
        ollama_response = self.ollama.generate_text(prompt, stop_sequences=stop, **kwargs)
        
        # Transform Ollama's response into BaseMessage
        # This transformation assumes BaseMessage can be constructed with text from Ollama
        # Adjust based on actual BaseMessage construction requirements
        response_message = BaseMessage(text=ollama_response)
        
        return response_message
    
    @property
    def ollama_instance(self):
        """Getter for the Ollama instance."""
        return self.ollama

    