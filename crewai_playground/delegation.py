import os
from crewai import Agent, Task, Crew, Process
from langchain.tools import DuckDuckGoSearchRun
from langchain_community.llms import Ollama
from langchain.agents import load_tools
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv
from llms import LLMFactory

load_dotenv

# ollama_llm = Ollama(model="openhermes", temperature=0.7)
# ollama_llm = LLMFactory().get_ollama_llm("stablelm-zephyr:3b-q6_K")
ollama_llm = LLMFactory().get_ollama_llm("olafgeibig/nous-hermes-2-mistral:7B-DPO-Q5_K_M")

search_tool = DuckDuckGoSearchRun()
human_tools = load_tools(["human"])
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools_list = [search_tool]

researcher = Agent(
  role='Senior researcher and Movies critique',
  goal='Craft compelling content on Movies',
  backstory="""Senior researcher and Movie critique. You will search the internet and find the newest upcoming best movies of 2024""",
  verbose=True,
  allow_delegation=False,
  tools=[wikipedia],
  llm=ollama_llm
)
writer = Agent(
  role='Tech Content Strategist',
  goal='Craft compelling content on the latest Movies',
  backstory="""You are a renowned Tech Content Strategist, known for your insightful
  and engaging articles on the latest Movies. With a deep understanding of
  the entertainment industry, you transform concepts into compelling narratives.""",
  verbose=True,
  allow_delegation=False,
  tools=[wikipedia],
  llm=ollama_llm
)

# Create tasks for your agents
# Being explicit on the task to ask for human feedback.
task1 = Task(
  description="""Conduct a comprehensive internet research for the upcoming best Movies of 2024.
  You MUST CHECK wikipedia to ONLY include Movies that have a release date in 2024 or 2025.
  Compile your findings in a detailed report.""",
	expected_output="A very lenghty report with all the movies found and their infromation.",
  agent=researcher
)

task2 = Task(
  description="""Using the insights from the researcher's report, develop an engaging article
   that highlights the most significant upcoming Movies of 2024.
  Your post should be informative yet accessible, catering to a entertainment-savvy audience.
  Aim for a narrative that captures entertains the audience. You must FORMAT text in Markdown language.""",
	expected_output="A complete full blog post using markdown of at least 5 paragraphs.",
	output_file="movies_report.md",
  agent=writer
)


project_crew = Crew(
    tasks=[task1, task2], # Tasks that that manager will figure out how to complete
    agents=[researcher, writer],
    full_output=False,
    manager_llm=ollama_llm, # The manager's LLM that will be used internally
    process=Process.hierarchical  # Designating the hierarchical approach
)

# Get your crew to work!
result = project_crew.kickoff()

#crew = Crew(agents=[agent], tasks=[task1, task2], full_output=True)

print("######### Final Results #############")
print(result)
print("######### Final Results #############")

