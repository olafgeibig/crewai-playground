import os
from crewai import Agent, Task, Crew, Process
from langchain.agents import load_tools
from dotenv import load_dotenv
from textwrap import dedent
from llm import LLM

load_dotenv()

# llm = Ollama(
#     model="nous-hermes-2-solar",
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
#     )
# llm = ChatAnyscale(
#     model="mistralai/Mixtral-8x7B-Instruct-v0.1",
#     # model="mistralai/Mistral-7B-Instruct-v0.1",
#     api_key=os.getenv('ANYSCALE_API_KEY'),
#     # base_url=os.getenv('ANYSCALE_BASE_URL'),
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
#     streaming=True
# )
# from langchain_community.llms import VertexAI
# llm = VertexAI(
#     model_name="gemini-pro",
#     location="us-central1",
#     project="mygpt-383514",
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
#     streaming=True
# )

llm = LLM().get_vertexai_llm("gemini-pro")

from langchain.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()
human_tool = load_tools(["human"])
from langchain_experimental.utilities import PythonREPL
python_tool = PythonREPL()

# Define your agents with roles and goals
manager = Agent(
  role='Project Manager',
  goal='Manages software projects and development teams.',
  backstory="""
  You are an experienced senior project manager with excellent managing skills. 
  You create project plans and you know how to lead a software development team.
  You have a modern, agile management style that empowers team members. You bring the
  skills of a team together. You create the project plan and assign tasks to the team
  members according to the plan. 
  """,
  verbose=True,
  allow_delegation=True,
  # tools=[search_tool]+human_tool,
  llm=llm
)
po = Agent(
  role='Product Owner',
  goal='Creates detailed specifications of a software product',
  backstory="""
  You are a very experienced product owner
  You transform the product vision of the human into technical specifications for developers.
  You discuss with the human and get his feedback and final approval of you specification,
  You ask the human for feedback to the project plan and get his final approval.
  """,
  verbose=True,
  allow_delegation=True,
  tools=[]+human_tool,
  llm=llm
)
dev = Agent(
  role='Developer',
  goal='Writes code',
  backstory="""
  You are a very experienced developer.
  You can code in many programming languages and you know all the best practices
  You write clean and readable code. You keep things simple and break down code into 
  reusable small units. You design code for testability.
  """,
  verbose=True,
  allow_delegation=False,
  # tools=[]+python_tool,
  llm=llm
)

product = "Calculator"
description = "A tool that can do basic calculations"

task1 = Task(
  description=dedent(f"""You need to understand the product vision of the human and transform it 
  into technical specifications for developers. Interact with the human and ask him questions
  if the product vision is not clear to you. Present your specification to the human for approval.
  If approved, hand over the sepcification to the project manager.
  Product name: {product}
  Product description: {description}"""),
  agent=po
)

task2 = Task(
  description=dedent(f"""You take the specification from the Product Owner and create a project plan
  for the development. You are breaking down the specifications into tasks for developers. Then
  you hand the tasks over to the developer.
  Product name: {product}
  Product description: {description}"""),
  agent=manager
)

task3 = Task(
  description=dedent(f"""You take the taks from the Project Manager and start working on them. You validate your code
  by trying to run it. If the excution is in error then fix the error and try again. If the code
  works properly, show the software to the Product Owner for review if it fulfills the specification.
  Product name: {product}
  Product description: {description}"""),
  agent=dev
)

task4 = Task(
  description=dedent(f"""You review the software produced by the developer. If it doesn't fulfil your
  specification then reject it and give the developer feedback how to improve it. If it fulfills
  the specification have it reviewed by the human and get his approval.
  Product name: {product}
  Product description: {description}"""),
  agent=po
)
# Instantiate your crew with a sequential process
crew = Crew(
  agents=[manager, po, dev],
  tasks=[task1, task2, task3, task4],
  verbose=1
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)
