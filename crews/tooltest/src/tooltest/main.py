from crewai import Agent, Task, Crew, Process
from crewai_tools import WebsiteSearchTool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import sys
from tooltest.tools.spider_tool import SpiderTool
from langtrace_python_sdk import langtrace

langtrace.init(
    api_key = '283c7ec7b220a4bb8d862640c44278639298e5fd17c2e4c240a2db9680aec9f7', 
    api_host="http://localhost:3000/api/trace",
    # write_spans_to_console=True,
)

load_dotenv()
inputs = {
    'topic': 'AI LLMs'
}

# ======== LLM Definitions ========================================

def deepseek_chat_llm():
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    return ChatOpenAI(
        model="deepseek-chat", 
        api_key=DEEPSEEK_API_KEY, 
        base_url="https://api.deepseek.com/beta",
        temperature=0.0
    )

def deepseek_coder_llm():
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    return ChatOpenAI(
        model="deepseek-coder", 
        api_key=DEEPSEEK_API_KEY, 
        base_url="https://api.deepseek.com/beta",
        temperature=0.0
    )

def gpt4o_llm():
    return ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0.7)

def sonnet_llm():
    return ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.7)

def groq_llm():
    return ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", temperature=0.7)

def groq_8b_llm():
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    return ChatOpenAI(
        model="llama-3.1-8b-instant",
        base_url="https://api.groq.com/openai/v1", 
        api_key=GROQ_API_KEY, 
        temperature=0.7)

def groq_70b_llm():
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    return ChatOpenAI(model="llama3-groq-70b-8192-tool-use-preview", api_key=GROQ_API_KEY, temperature=0.7)

def google_gemini_flash():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# ======== LLM Abstractions

def default_llm():
    return deepseek_chat_llm()

def simple_llm():
    return deepseek_chat_llm()

def reasoning_llm():
    return gpt4o_llm()

def smart_llm():
    return sonnet_llm()

def coding_llm():
    return deepseek_coder_llm()

# ======== Agent Definitions ========================================

def create_simple_agent(llm):
    return Agent(
        role='Websearch Agent',
        goal='Research the content of a webpages',
        backstory='I am a web researcher.',
        verbose=True,
        llm=llm,
        tools=[ WebsiteSearchTool(website="https://www.heise.de/news/Weltweiter-IT-Ausfall-Flughaefen-Banken-und-Geschaefte-betroffen-9806343.html")]
    )

def create_spider_agent(llm):
    return Agent(
        role='Spider Agent',
        goal='Crawl a website and extract the essential information',
        backstory='An expert web researcher that uses the web extremely well. You know how to use a spider to crawl a website.',
        verbose=True,
        llm=llm,
        tools=[SpiderTool()],
        max_execution_time=1800, 
    )

# ======== Task Definitions ========================================

def create_spider_task(agent):      
    return Task(
        description="""
        Use the spider to get the website's content and then extract the relevant information from it.

        Crawl 'https://python.langchain.com/v0.1/docs/integrations/tools/'

        with the following options:
        - limit: 2
        - depth: 1

        The goal is to create a documentation of the tools in markdown format. The documentation 
        must contain the following information for each tool:
        1. Name of the tool
        2. Description
        3. Arguments and their meaning
        4. Requirements, e.g. needed API keys, needed libraries to import, config files, etc. 
        5. Possible use cases for an AI agent for this tool
        """,
        agent=agent,
        expected_output="""A list of tools in markdown format as follows:
        # Tool Name
        ## Description
        Description of the tool
        ## Usage
        ### Tool
        Tool module name
        ### Arguments
        Bulleted list of arguments and their meaning
        ## Requirements
        Bulleted list of requirements, e.g. needed API keys, needed python packages to install, config files, etc.
        ## Use Cases
        Bulleted list of use cases for agents
        """,
        output_file="langchain_tools.md"
    )

# ======== Crew Definition ========================================

def crew() -> Crew:
    llm = default_llm()
    spider_agent = create_spider_agent(llm)
    return Crew(
        agents=[
            spider_agent
        ],
        tasks=[
            create_spider_task(spider_agent)
        ],
        process=Process.sequential,
        verbose=True,
        cache=True,
    )

# ======== CrewAI Stuff ========================================

def run():
    """
    Run the crew.
    """
    # with langtrace.trace("CrewAI Task Execution"):
    crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    try:
        crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    try:
        crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")
