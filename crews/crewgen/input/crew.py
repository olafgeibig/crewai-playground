from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew
from crewai_tools import FileReadTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

@CrewBase
class CrewGenCrew():
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self) -> None:
        load_dotenv()
        # self.llm=ChatOpenAI(model="gpt-4o", temperature=0.7)
        DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
        self.llm = ChatOpenAI(
            model="deepseek-chat", 
            api_key=DEEPSEEK_API_KEY, 
            base_url="https://api.deepseek.com/beta",
            temperature=0.0
        )

    @agent
    def create_example_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['example_agent'],
            llm=self.llm,
            verbose=True,
            memory=True,
            allow_delegation=True,
            tools=[
                FileReadTool()
            ] 
        )
    
    @task
    def example_task(self) -> Task:
        return Task(
            config=self.tasks_config['example_task'],
            agent=self.create_example_agent(),
            output_file='result.md'
        )
	
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )