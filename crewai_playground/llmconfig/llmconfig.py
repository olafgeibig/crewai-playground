import os
from dotenv import load_dotenv
import yaml
from langchain.chat_models import ChatOpenAI  # Assuming ChatOpenAI is in langchain.chat_models

class LLMFactory:
    def __init__(self, config_path):
        load_dotenv()
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def get_llm(self, llm_path):
        keys = llm_path.split('.')
        config = self.config
        for key in keys:
            config = config.get(key, {})
        if not config:
            raise ValueError(f"Configuration for {llm_path} not found")

        class_name = config.get('class')
        if not class_name:
            raise ValueError(f"Class not specified for {llm_path}")

        # Assuming all classes are in langchain.chat_models and named as in the 'class' field
        llm_class = getattr(langchain.chat_models, class_name)
        # Update the parameters, resolving any environment variables for api_key
        parameters = {k: (os.getenv(v) if k == 'api_key' else v) for k, v in config.items() if k != 'class'}
        instance = llm_class(**parameters)
        return instance
