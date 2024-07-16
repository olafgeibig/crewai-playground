from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Note that the docstrings here are crucial, as they will be passed along
# to the model along with the class name.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")
# model = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="foo")

model = ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key="b4884adff34ff7665f0ca99cef4891db99ef6a33241f61f14bc6d4ff7609c542",
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
)
llm_with_tools = model.bind_tools([Joke])
llm_with_tools.invoke("what's 3 * 12")

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers.openai_tools import PydanticToolsParser

tool_chain = llm_with_tools | PydanticToolsParser(tools=[Joke])
print(tool_chain.invoke("what's 3 * 12"))

parser = JsonOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

chain.invoke({"query": joke_query})