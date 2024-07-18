import os
from crewai import Agent, Task, Crew, Process
from langchain.agents import load_tools
from dotenv import load_dotenv
from textwrap import dedent
from attic.utils.llms import LLMFactory
from langchain.tools import Tool
from attic.tools.attic.code_interpreter_tools import CodeInterpreterTool

llm = LLMFactory().get_ollama_llm("neuralbeagle-agent")
# llm = LLMFactory().get_anyscale_llm("mistralai/Mixtral-8x7B-Instruct-v0.1")
# llm = LLMFactory().get_together_ai_llm("NousResearch/Nous-Hermes-2-Yi-34B")
# llm = LLMFactory().get_openai_llm(model="foo",api_key="foo", base_url="https://cbszkksghhaox2-5000.proxy.runpod.net/v1")
llm.temperature=0.1

load_dotenv()

# from langchain_community.tools import DuckDuckGoSearchRun
# search_tool = DuckDuckGoSearchRun()
# human_tool = load_tools(["human"])

# from langchain_experimental.utilities import PythonREPL
# pythonrepl = PythonREPL()
# python_tool = Tool(
#     name="python_repl",
#     description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
#     func=pythonrepl.run,
# )

# ==== Tool Definitions ============================================================

# ==== Agent Definitions ============================================================

dev = Agent(
  role='Developer',
  goal='Write and run code',
  backstory="""\
  You are a very experienced developer.
  You can code in many programming languages and you know all the best practices
  You write clean and readable code. You keep things simple and break down code into 
  reusable small units. You design code for testability.
  """,
  verbose=True,
  allow_delegation=False,
  llm=llm
)

# codeInterpreterTool = CodeInterpreterTool()

tester = Agent(
  role='Tester',
  goal='Test and run code',
  backstory="""\
  You are a very experienced software tester.
  You follow the instruction thoroughly. You execute the pure code. When running the code do not embed it in Markdown and do not wrap it in quotes.
  If you get the code from the developer in markdown, then extract the pure code from it.
  Check if the code is formatted and indented properly. If not correct it.
  """,
  verbose=True,
  allow_delegation=False,
  tools=[
      CodeInterpreterTool()
      ],
  llm=llm
)

# ==== Task Definitions ============================================================

task_dev = Task(
  description=dedent("""
  Develop python software that determines today's date. You don't need a tool, only write the code. When you are done, send the code to the tester for verification."""),
  agent=dev
)

task_test = Task(
  description=dedent("""
  Test and run this python code "from datetime import date; todays_date = date.today(); print("Today's date is:", todays_date)" using the run_python execution tool. Extract the code from the context and run it. 
  Do not just call a function or put the context into the execution environment. Extract the pure code first. Do not use Markdown.
  Check if the code is formatted and indented properly. NEVER use markdown.
"""),
  agent=tester
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[tester],
  tasks=[task_test],
  verbose=1
)

# Get your crew to work!
result = crew.kickoff()
# result = CodeInterpreterTool.run_python("print('hello')")
print("######################")
print(result)
