from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew
from crewai_tools import ScrapeWebsiteTool, FileReadTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

@CrewBase
class CrewGenCrew():
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self) -> None:
        load_dotenv()
        # llm=ChatOpenAI(model="gpt-4o", temperature=0.7)
        DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
        self.llm = ChatOpenAI(
            model="deepseek-chat", 
            api_key=DEEPSEEK_API_KEY, 
            base_url="https://api.deepseek.com/beta",
            temperature=0.0
        )

    @agent
    def create_manager_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['manager_agent'],
            llm=self.llm,
            verbose=True,
            memory=True,
            allow_delegation=False,      
        )

    @agent
    def create_agent_designer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['agent_designer'],
            llm=self.llm,
            verbose=True,
            memory=True,
            allow_delegation=False,
            tools=[
                FileReadTool(),
            ]
        )

    @agent
    def create_task_designer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['task_designer'],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def create_coder_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['agent_coder'],
            llm=self.llm,
            verbose=True,
            memory=True,
            allow_delegation=False,
            tools=[
                FileReadTool(),
            ]
        )

    @task
    def team_task(self) -> Task:
        return Task(
            config=self.tasks_config['team_concept'],
            agent=self.create_manager_agent(),
            output_file='./output/team_concept.md'
        )

    @task
    def agent_design_task(self) -> Task:
        return Task(
            config=self.tasks_config['agent_definition'],
            agent=self.create_agent_designer_agent(),
            output_file='./output/agents.yaml'
        )

    @task
    def task_design_task(self) -> Task:
        return Task(
            config=self.tasks_config['task_definition'],
            agent=self.create_task_designer_agent(),
            output_file='./output/tasks.yaml'
        )

    @task
    def crew_code_task(self) -> Task:
        return Task(
            config=self.tasks_config['crew_code'],
            agent=self.create_coder_agent(),
            output_file='./output/crew.py'
        )
	
    @task
    def agents_improvement_task(self) -> Task:
        return Task(
            config=self.tasks_config['tool_usage'],
            agent=self.create_coder_agent(),
            output_file='./output/crew.py'
        )
	
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            cache=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )