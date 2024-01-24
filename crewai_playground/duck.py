from crewai import Agent, Task, Crew, Process
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import load_tools
from llms import LLMFactory
import os
from dotenv import load_dotenv
load_dotenv()

#llm = LLMFactory().get_anyscale_llm("mistralai/Mixtral-8x7B-Instruct-v0.1")
llm = LLMFactory().get_ollama_llm(model="neuralbeagle-agent", base_url="http://192.168.2.50:11434")
# llm = LLMFactory().get_azure_llm("gpt-4-1106-Preview", "gpt-4")
# llm = LLMFactory().get_openai_llm(model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key=os.getenv("DEEPINFRA_API_KEY"), base_url="https://api.deepinfra.com/v1/openai")
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
  actionable insights. You ask the human for feedback if the result of a research is rekevant for the report.""",
  verbose=True,
  allow_delegation=False,
  # Passing human tools to the agent
  tools=[search_tool]+human_tool,
  llm=llm
)
writer = Agent(
  role='Tech Content Strategist',
  goal='Craft compelling content on tech advancements',
  backstory="""You are a renowned Tech Content Strategist, known for your insightful
  and engaging articles on technology and innovation. With a deep understanding of
  the tech industry, you transform complex concepts into compelling narratives.""",
  verbose=True,
  allow_delegation=True,
  llm=llm
)

# Create tasks for your agents
# Being explicit on the task to ask for human feedback.
task1 = Task(
  description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.
  Compile your findings in a detailed report. 
  Make sure to check with the human if the draft is good before returning your Final Answer.
  Your final answer MUST be a full analysis report""",
  agent=researcher
)

task2 = Task(
  description="""Using the insights from the researcher's report, develop an engaging blog
  post that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Aim for a narrative that captures the essence of these breakthroughs and their
  implications for the future. 
  Your final answer MUST be the full blog post of at least 3 paragraphs.""",
  agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=0
)

# Get your crew to work!
result = crew.kickoff()

print(result)
