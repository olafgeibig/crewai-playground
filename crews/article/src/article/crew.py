from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from crewai_tools import SerperDevTool, WebsiteSearchTool
from llms import LLMFactory

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
		self.llm = LLMFactory().get_together_ai_llm("mistralai/Mistral-7B-Instruct-v0.2")

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
			agent=self.researcher(),
			output_file='research_findings.md'
		)

	@task
	def conversation_task(self) -> Task:
		return Task(
			config=self.tasks_config['conversation_task'],
			agent=self.conversation_simulator(),
			output_file='conversation_transcript.md'
		)

	@task
	def outline_task(self) -> Task:
		return Task(
			config=self.tasks_config['outline_task'],
			agent=self.outline_creator(),
			output_file='article_outline.md'
		)

	@task
	def writing_task(self) -> Task:
		return Task(
			config=self.tasks_config['writing_task'],
			agent=self.article_writer(),
			output_file='wikipedia_article_draft.md'
		)

	@task
	def revision_task(self) -> Task:
		return Task(
			config=self.tasks_config['revision_task'],
			agent=self.revision_expert(),
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