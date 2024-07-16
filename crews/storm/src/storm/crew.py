from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from crewai_tools import SerperDevTool, WebsiteSearchTool
from storm.llms import LLMFactory


@CrewBase
class StormCrew():
	"""Storm crew"""
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	def __init__(self) -> None:
		"""
		Initializes the StormCrew object.

		This method initializes the StormCrew object by creating an instance of the WebsiteSearchTool class and configuring it
		using the rag_config.yaml file. It then sets the self.tools attribute to a list containing an instance of the
		SerperDevTool class and the website_search_tool instance. It also sets the self.llm attribute to an instance of the
		Ollama language model obtained from the LLMFactory class using the specified model name. Finally, it sets the
		self.max_rpm attribute to 100.

		Parameters:
		    None

		Returns:
		    None
		"""
		website_search_tool.from_embedchain("./rag_config.yaml")
		self.tools = [
			SerperDevTool(),
			website_search_tool
		]
		# self.llm = LLMFactory().get_anthropic_llm("claude-3-haiku-20240307")
		self.llm = LLMFactory().get_ollama_llm("olafgeibig/nous-hermes2pro:Q5_K_M")
		self.max_rpm=100


	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			tools=self.tools,
			llm=self.llm,
			max_rpm=self.max_rpm,
			verbose=True
		)

	@agent
	def conversation_simulator(self) -> Agent:
		return Agent(
			config=self.agents_config['conversation_simulator'],
			llm=self.llm,
			max_rpm=self.max_rpm,
			verbose=True
		)

	@agent
	def outline_creator(self) -> Agent:
		return Agent(
			config=self.agents_config['outline_creator'],
			llm=self.llm,
			max_rpm=self.max_rpm,
			verbose=True
		)

	@agent
	def article_writer(self) -> Agent:
		return Agent(
			config=self.agents_config['article_writer'],
			tools=self.tools,
			llm=self.llm,
			max_rpm=self.max_rpm,
			verbose=True
		)

	@agent
	def revision_expert(self) -> Agent:
		return Agent(
			config=self.agents_config['revision_expert'],
			tools=self.tools,
			llm=self.llm,
			max_rpm=self.max_rpm,
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