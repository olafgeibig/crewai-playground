from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew, llm
from crewai_tools import FileReadTool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

@CrewBase
class GeneratedCrew():
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self) -> None:
        load_dotenv()

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
        return ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.7)

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
    def example_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['example_agent'],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            cache=True,
            tools=[] 
        )
    

    # ======== Task Definitions ========================================
    
    @task
    def example_task(self) -> Task:
        return Task(
            config=self.tasks_config['example_task'],
            output_file='result.md',
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
            # memory=True,
            cache=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )