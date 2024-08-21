from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew, llm
from crewai_tools import (
    FileReadTool, DirectorySearchTool, DirectoryReadTool
)
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

@CrewBase
class DocumentationGenerationCrew():
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
    
    @llm
    def groq_llm(self):
        return ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", temperature=0.7)
    
    @llm 
    def groq_8b_llm(self):
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        return ChatOpenAI(
            model="llama-3.1-8b-instant",
            base_url="https://api.groq.com/openai/v1", 
            api_key=GROQ_API_KEY, 
            temperature=0.7)

    @llm 
    def groq_70b_llm(self):
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        return ChatOpenAI(model="llama3-groq-70b-8192-tool-use-preview", api_key=GROQ_API_KEY, temperature=0.7)

    @llm
    def google_gemini_flash(self):
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    
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
    def repository_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config['repository_analyzer'],
            verbose=True,
            allow_delegation=False,
            cache=True,
            tools=[FileReadTool(), DirectoryReadTool(), DirectorySearchTool()],
        )

    @agent
    def information_extractor(self) -> Agent:
        return Agent(
            config=self.agents_config['information_extractor'],
            verbose=True,
            allow_delegation=False,
            cache=True,
            tools=[FileReadTool()],
            max_execution_time=1800,           
        )

    @agent
    def use_case_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['use_case_specialist'],
            verbose=True,
            allow_delegation=False,
            cache=True,
            tools=[FileReadTool(), DirectoryReadTool(directory="./output")]
        )

    @agent
    def documentation_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['documentation_writer'],
            verbose=True,
            allow_delegation=False,
            cache=True,
            tools=[FileReadTool(), DirectoryReadTool(directory="./output")]
        )

    @agent
    def quality_assurance_reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config['quality_assurance_reviewer'],
            verbose=True,
            allow_delegation=False,
            cache=True,
            tools=[FileReadTool(), DirectoryReadTool(directory="./output")]
        )

    @agent
    def documentation_compiler(self) -> Agent:
        return Agent(
            config=self.agents_config['documentation_compiler'],
            verbose=True,
            allow_delegation=False,
            cache=True,
            tools=[FileReadTool(), DirectoryReadTool(directory="./output")]
        )

    @agent
    def human_liaison(self) -> Agent:
        return Agent(
            config=self.agents_config['human_liaison'],
            verbose=True,
            allow_delegation=False,
            cache=True,
            tools=[],
        )

    # ======== Task Definitions ========================================
    
    @task
    def repository_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['repository_analysis_task'],
            output_file='./output/01_repository_analysis_result.md',
            cache=True,
        )

    @task
    def information_extraction_task(self) -> Task:
        return Task(
            config=self.tasks_config['information_extraction_task'],
            output_file='./output/02_information_extraction_result.md',
            cache=True,
        )

    @task
    def use_case_identification_task(self) -> Task:
        return Task(
            config=self.tasks_config['use_case_identification_task'],
            output_file='./output/03_use_cases.md',
            cache=True,
        )

    @task
    def documentation_writing_task(self) -> Task:
        return Task(
            config=self.tasks_config['documentation_writing_task'],
            output_file='./output/04_tool_documentation.md',
            cache=True,
        )

    @task
    def quality_assurance_task(self) -> Task:
        return Task(
            config=self.tasks_config['quality_assurance_task'],
            output_file='./output/05_qa_review.md',
            cache=True,
        )

    @task
    def documentation_compilation_task(self) -> Task:
        return Task(
            config=self.tasks_config['documentation_compilation_task'],
            output_file='./output/06_final_documentation.md',
            cache=True,
            human_input=True
        )

    # @task
    # def human_review_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['human_review_task'],
    #         output_file='./output/07_human_review_report.md',
    #         cache=True,
    #     )

    # ======== Crew Definition ========================================

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            cache=True,
        )