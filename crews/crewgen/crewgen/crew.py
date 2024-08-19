from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, FileReadTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import yaml

@CrewBase
class CrewGenCrew():
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self) -> None:
        self.tools = [
            SerperDevTool()
        ]
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
            verbose=True
        )

    @agent
    def create_agent_specialist_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['agent_agent'],
            llm=self.llm,
            verbose=True
        )

    @agent
    def create_task_specialist_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['task_agent'],
            llm=self.llm,
            verbose=True
        )

    @task
    def team_task(self) -> Task:
        return Task(
            config=self.tasks_config['team_task'],
            agent=self.create_manager_agent(),
            output_file='team_composition.md'
        )

    @task
    def agent_task(self) -> Task:
        return Task(
            config=self.tasks_config['agent_task'],
            agent=self.create_agent_specialist_agent(),
            output_file='agent_definitions.yaml'
        )
