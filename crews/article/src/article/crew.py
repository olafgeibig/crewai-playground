from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, WebsiteSearchTool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

@CrewBase
class ArticleCrew():
	"""Article crew"""
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	def __init__(self) -> None:
		self.tools = [
			SerperDevTool(),
			WebsiteSearchTool()
		]
		# load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))
		load_dotenv()
		self.llm = ChatGroq(model_name="llama3-groq-8b-8192-tool-use-preview")

	@agent
	def create_researcher_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			tools=self.tools,
			llm=self.llm,
			verbose=True
		)

	@agent
	def create_conversation_simulator_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['conversation_simulator'],
			llm=self.llm,
			verbose=True
		)

	@agent
	def create_outline_creator_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['outline_creator'],
			llm=self.llm,
			verbose=True
		)

	@agent
	def create_article_writer_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['article_writer'],
			tools=self.tools,
			llm=self.llm,
			verbose=True
		)

	@agent
	def create_revision_expert_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['revision_expert'],
			tools=self.tools,
			llm=self.llm,
			verbose=True
		)

	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
			agent=self.create_researcher_agent(),
			output_file='research_findings.md'
		)

	@task
	def conversation_task(self) -> Task:
		return Task(
			config=self.tasks_config['conversation_task'],
			agent=self.create_conversation_simulator_agent(),
			output_file='conversation_transcript.md'
		)

	@task
	def outline_task(self) -> Task:
		return Task(
			config=self.tasks_config['outline_task'],
			agent=self.create_outline_creator_agent(),
			output_file='article_outline.md'
		)

	@task
	def writing_task(self) -> Task:
		return Task(
			config=self.tasks_config['writing_task'],
			agent=self.create_article_writer_agent(),
			output_file='wikipedia_article_draft.md'
		)

	@task
	def revision_task(self) -> Task:
		return Task(
			config=self.tasks_config['revision_task'],
			agent=self.create_revision_expert_agent(),
			output_file='final_wikipedia_article.md'
		)
	
	@crew
	def crew(self) -> Crew:
		"""Creates the Article crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=2,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)