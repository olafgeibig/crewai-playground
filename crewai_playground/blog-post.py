from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from llms import LLMFactory

# ollama_solar_llm = Ollama(model="solar:latest")
# ollama_mistral_llm = Ollama(model="dolphin-mistral:latest")

# ollama_solar_llm = LLMFactory().get_openai_llm(model="stablelm-zephyr:3b", base_url="http://localhost:11434/v1/", api_key="foo")
ollama_solar_llm = LLMFactory().get_ollama_llm("nous-hermes-2-mistral:7B-DPO-Q5_K_M")
ollama_mistral_llm = ollama_solar_llm


MAX_ITER = 15

search_tool = DuckDuckGoSearchRun()

sequential_process = Process.sequential

# Define your agents with roles and goals
researcher = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge developments in AI and data science',
  backstory="""You work at a leading tech think tank.
  Your expertise lies in identifying emerging trends.
  You have a knack for dissecting complex data and presenting
  actionable insights.""",
  verbose=True,
  allow_delegation=False,
  tools=[search_tool],
  llm=ollama_solar_llm,
  max_iter=MAX_ITER
)
writer = Agent(
  role='Tech Content Strategist',
  goal='Craft compelling content on tech advancements',
  backstory="""You are a renowned Content Strategist, known for
  your insightful and engaging articles.
  You transform complex concepts into compelling narratives.""",
  verbose=True,
  allow_delegation=True,
  llm=ollama_mistral_llm,
  max_iter=MAX_ITER
)

# Create tasks for your agents
task1 = Task(
  description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.""",
	expected_output="A very lenghty report with all the insights and trends found.",
  agent=researcher
)

task2 = Task(
  description="""Using the insights provided, develop an engaging blog
  post that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Make it sound cool, avoid complex words so it doesn't sound like AI.""",
	expected_output="The full blog post using markdown with title and section of at least 6 paragraphs.",
  agent=writer,
	output_file="blog_post.md"
)


crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=1,
  process=sequential_process
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)