from crewai import Agent
from tools.browser_tools import BrowserTools
from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools
from crewai import Task
from textwrap import dedent
from llms import LLMFactory
import os
from dotenv import load_dotenv
load_dotenv()

llm = LLMFactory().get_ollama_llm("neuralbeagle-agent")

research_agent = Agent(
    role='Research topics and summarize',
    goal="""You review travel plans and iternaries that shall be proposed to customers by co-workers.
    If you are not sure about the proposals, you do your own research. If you are not happy, reject
    the proposal and the co-workers shall improve it.
    """,
    backstory="""Super experienced traveler who knows all tourist destinations from his own experience.
    You have traveled the whole world and you can judge if a proposed iternary
    is good. You can judge if an activity, restaurant or sight is worth to do.
    """,
    verbose=True,
    llm=llm,
    tools=[
    BrowserTools.scrape_and_summarize_website,
    SearchTools.search_internet,
    ]
)

Task(description=dedent(f"""
        Analyze and select the best places for the trip based 
        on specific criteria such as weather patterns, seasonal
        events, and travel costs. This task involves comparing
        multiple locations, considering factors like current weather
        conditions, upcoming cultural or seasonal events, and
        overall travel expenses. 
        
        Your final answer must be a detailed
        report on the chosen places, and everything you found out
        about it, including the actual flight costs, weather 
        forecast and attractions.
        {self.__tip_section()}

        Traveling from: {origin}
        City Options: {cities}
        Trip Date: {range}
        Traveler Interests: {interests}
        Traveler hints: {hints}
      """),
      agent = research_agent)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=0
)

# Get your crew to work!
result = crew.kickoff()

print(result)