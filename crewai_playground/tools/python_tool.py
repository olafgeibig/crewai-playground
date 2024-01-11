from langchain.tools import tool
from interpreter import interpreter

# interpreter.offline = True
interpreter.verbose = True
interpreter.llm.model = "ollama/nous-hermes-2-solar:latest"
interpreter.llm.api_base = "http://127.0.0.1:11434/"

interpreter.chat("What is 24 + 23?")
print(interpreter.messages)