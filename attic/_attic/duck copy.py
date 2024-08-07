from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import load_tools
from llms import LLMFactory
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
# llm = LLMFactory().get_anyscale_llm("mistralai/Mixtral-8x7B-Instruct-v0.1")
llm = LLMFactory().get_ollama_llm(model="olafgeibig/nous-hermes2pro:Q5_K_M") #, base_url="http://192.168.2.50:11434")
# llm = LLMFactory().get_openai_llm(model="phi-2-openhermes-2.5:2.7B-Q5_K_M", base_url="http://localhost:11434/v1/", api_key="foo")
# llm = LLMFactory().get_azure_llm("gpt-4-1106-Preview", "gpt-4")
# llm = LLMFactory().get_openai_llm(model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key=os.getenv("DEEPINFRA_API_KEY"), base_url="https://api.deepinfra.com/v1/openai")
# llm = LLMFactory().get_google_ai_llm("gemini-pro")
# llm = LLMFactory().get_together_ai_llm("teknium/OpenHermes-2p5-Mistral-7B")
# llm = LLMFactory().get_openai_llm(model="mistralai/Mistral-7B-Instruct-v0.1", api_key=os.getenv("TOGETHERAI_API_KEY"), base_url=os.getenv("TOGETHERAI_BASE_URL"))
# llm = LLMFactory().get_anyscale_llm("mistralai/Mistral-7B-Instruct-v0.1")
from langchain_experimental.llms.ollama_functions import OllamaFunctions
# llm_func =  OllamaFunctions(model="neuralbeagle-agent:latest")
# llm_func = LLMFactory().get_openai_llm(model="gpt-4-0125-preview")
# llm = llm_func
llm_func = llm
llm.temperature = 0.1

from langchain_community.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()
human_tool = load_tools(["human"])

# Define your agents with roles and goals

researcher = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge developments in AI and data science in',
  backstory="""You are a Senior Research Analyst at a leading tech think tank.
  Your expertise lies in identifying emerging trends and technologies in AI and
  data science. You have a knack for dissecting complex data and presenting
  actionable insights. You ask the human for feedback if the result of a research is relevant for the report.""",
  verbose=True,
  allow_delegation=False,
  # Passing human tools to the agent
  tools=[search_tool],
  llm=llm,
  function_calling_llm=llm_func
)
writer = Agent(
  role='Tech Content Strategist',
  goal='Craft compelling content on tech advancements',
  backstory="""You are a renowned Tech Content Strategist, known for your insightful
  and engaging articles on technology and innovation. With a deep understanding of
  the tech industry, you transform complex concepts into compelling narratives.""",
  verbose=True,
  allow_delegation=True,
  llm=llm,
  function_calling_llm=llm_func
)

# Create tasks for your agents
@task
def task1(self) -> Task:
  return Task(
  description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.
  Compile your findings in a detailed report. 
  Make sure to check with the human if the draft is good before returning your Final Answer.
  Your final answer MUST be a full analysis report.""",
  agent=researcher
)

@task
def task2(self) -> Task:
  return Task(
  description="""Using the insights from the researcher's report, develop an engaging blog
  post that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Aim for a narrative that captures the essence of these breakthroughs and their
  implications for the future. 
  Your final answer MUST be the full blog post of at least 3 paragraphs.""",
  agent=writer
)

@crew
def crew(self) -> Crew:
  return Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=1
)

# Get your crew to work!
result = crew.kickoff()

print(result)
