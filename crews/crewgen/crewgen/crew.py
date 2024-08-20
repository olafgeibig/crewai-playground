from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew, llm
from crewai_tools import ScrapeWebsiteTool, FileReadTool, DirectoryReadTool, FileWriterTool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

@CrewBase
class CrewGenCrew():
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self) -> None:
        load_dotenv()
        # self.agent_coder = agent_coder(self)


    # ======== LLM Definitions ========================================

    @llm
    def deepseek_chat_llm(self):
        DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
        return ChatOpenAI(
            model="deepseek-chat", 
            api_key=DEEPSEEK_API_KEY, 
            base_url="https://api.deepseek.com/beta",
            temperature=0.0
        )
    
    @llm
    def deepseek_coder_llm(self):
        DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
        return ChatOpenAI(
            model="deepseek-coder", 
            api_key=DEEPSEEK_API_KEY, 
            base_url="https://api.deepseek.com/beta",
            temperature=0.0
        )
    
    @llm
    def gpt4o_llm(self):
        return ChatOpenAI(model="gpt-4o-latest", temperature=0.7)
    
    @llm 
    def sonnet_llm(self):
        return ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.7, max_tokens=8192)

    # ======== LLM Abstractions

    @llm
    def default_llm(self):
        return self.deepseek_chat_llm()
    
    @llm
    def simple_llm(self):
        return self.deepseek_chat_llm()
    
    @llm
    def reasoning_llm(self):
        return self.gpt4o_llm()
    
    @llm
    def smart_llm(self):
        return self.sonnet_llm()
    
    @llm
    def coding_llm(self):
        return self.deepseek_coder_llm()
    

    # ======== Agent Definitions ========================================

    @agent
    def crew_designer(self) -> Agent:
        return Agent(
            config=self.agents_config['crew_designer'],
            verbose=True,
            # memory=True,
            cache=True,
            allow_delegation=False,      
        )

    @agent
    def agent_designer(self) -> Agent:
        return Agent(
            config=self.agents_config['agent_designer'],
            verbose=True,
            # memory=True,
            cache=True,
            allow_delegation=False,
            tools=[
                FileReadTool()
            ]
        )

    @agent
    def task_designer(self) -> Agent:
        return Agent(
            config=self.agents_config['task_designer'],
            verbose=True,
            # memory=True,
            cache=True,
            allow_delegation=False,
            tools=[
                FileReadTool()
            ]
        )

    @agent
    def agent_coder(self) -> Agent:
        return Agent(
            config=self.agents_config['agent_coder'],
            verbose=True,
            memory=True,
            allow_delegation=False,
            cache=True,
            tools=[
                FileReadTool(),
                # DirectoryReadTool(directory="./output"),
                # FileWriterTool()
            ]
        )


    # ======== Task Definitions ========================================

    @task
    def crew_concept(self) -> Task:
        return Task(
            config=self.tasks_config['crew_concept'],
            output_file='./output/crew_concept.md',
            cache=True,
        )

    @task
    def agent_definition(self) -> Task:
        return Task(
            config=self.tasks_config['agent_definition'],
            output_file='./output/agents.yaml',
            cache=True,
        )

    @task
    def task_definition(self) -> Task:
        return Task(
            config=self.tasks_config['task_definition'],
            output_file='./output/tasks.yaml',
            cache=True,
        )
    
    @task
    def crew_code(self) -> Task:
        return Task(
            config=self.tasks_config['crew_code'],
            output_file='./output/crew.py',
            cache=True,
        )
	
    @task
    def tool_usage(self) -> Task:
        return Task(
            config=self.tasks_config['tool_usage'],
            output_file='./output/crew.py',
            cache=True,
        )
	

    # ======== Crew Definition ========================================

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            cache=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
            # memory=True,  # Enable memory usage for enhanced task execution
            # manager_llm=self.llm,  # Optional: explicitly set a specific agent as manager instead of the manager_llm
            # planning=True,  # Enable planning feature for pre-execution strategy
        )