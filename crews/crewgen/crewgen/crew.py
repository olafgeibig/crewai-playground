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
        llm = ChatOpenAI(
            model="deepseek-chat", 
            api_key=DEEPSEEK_API_KEY, 
            base_url="https://api.deepseek.com/beta",
            temperature=0.0
        )
