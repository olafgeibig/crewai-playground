import os
from crewai import Agent, Task, Crew, Process
from langchain.chat_models import ChatAnyscale
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.agents import load_tools
from dotenv import load_dotenv

load_dotenv()

# llm = Ollama(
#     model="nous-capybara",
#     base_url="https://e993-35-229-104-68.ngrok-free.app",
#     # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
#     )
llm = ChatAnyscale(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    # model="mistralai/Mistral-7B-Instruct-v0.1",
    api_key=os.getenv('ANYSCALE_API_KEY'),
    # base_url=os.getenv('ANYSCALE_BASE_URL'),
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    streaming=True
)
# from langchain_community.llms import VertexAI
# llm = VertexAI(
#     model_name="gemini-pro",
#     location="us-central1",
#     project="mygpt-383514",
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
#     streaming=True
# )


from langchain.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()
human_tool = load_tools(["human"])

# Define your agents with roles and goals
researcher = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge developments in AI and data science in',
  backstory="""You are a Senior Research Analyst at a leading tech think tank.
  Your expertise lies in identifying emerging trends and technologies in AI and
  data science. You have a knack for dissecting complex data and presenting
  actionable insights. You ask the human for feedback if the result of a research is rekevant for the report.""",
  verbose=True,
  allow_delegation=False,
  # Passing human tools to the agent
  tools=[search_tool]+human_tool,
  llm=llm
)
writer = Agent(
  role='Tech Content Strategist',
  goal='Craft compelling content on tech advancements',
  backstory="""You are a renowned Tech Content Strategist, known for your insightful
  and engaging articles on technology and innovation. With a deep understanding of
  the tech industry, you transform complex concepts into compelling narratives.""",
  verbose=True,
  allow_delegation=True,
  llm=llm
)

# Create tasks for your agents
# Being explicit on the task to ask for human feedback.
task1 = Task(
  description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.
  Compile your findings in a detailed report. 
  Make sure to check with the human if the draft is good before returning your Final Answer.
  Your final answer MUST be a full analysis report""",
  agent=researcher
)

task2 = Task(
  description="""Using the insights from the researcher's report, develop an engaging blog
  post that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Aim for a narrative that captures the essence of these breakthroughs and their
  implications for the future. 
  Your final answer MUST be the full blog post of at least 3 paragraphs.""",
  agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=0
)

# Get your crew to work!
result = crew.kickoff()

print("-------------------------------------")
print(result)
