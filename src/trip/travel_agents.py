from crewai import Agent
from src.tools.browser_tools import BrowserTools
from src.tools.calculator_tools import CalculatorTools
from src.tools.search_tools import SearchTools

from utils.llms import LLMFactory
import os
from dotenv import load_dotenv
load_dotenv()

# llm = LLMFactory().get_ollama_llm("neuralbeagle-agent")
llm = LLMFactory().get_together_ai_llm("meta-llama/Llama-3-70b-chat-hf")
# llm = LLMFactory().get_openai_llm(model="foo",api_key="foo", base_url="https://cbszkksghhaox2-5000.proxy.runpod.net/v1")

class TravelAgents():
  def travel_reviewer(self):
    return Agent(
      role='Travel Reviewer',
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

  def local_expert(self):
    return Agent(
        role='Local Expert at this destination',
        goal="""Provide the BEST insights about the selected destination. Check the security information for the destination.
        Find all the secret tipps and amazing experiences off the beaten path.
        """,
        backstory="""A knowledgeable local guide with extensive information
        about the destination, it's attractions and customs. You know all the secrets that most tourists don't know.
        You know the hangouts of the cool locals. You know the best activiies, security situation and the weather.""",
        llm=llm,
        tools=[
            SearchTools.search_internet,
            BrowserTools.scrape_and_summarize_website,
            SearchTools.search_news
        ],
        verbose=True)
    
  def travel_concierge(self):
    return Agent(
        role='Amazing Travel Concierge',
        goal="""Create the most amazing travel itineraries with budget and 
        packing suggestions for the city. You ask the local experts for best tipps.""",
        backstory="""Specialist in travel planning and logistics with decades of experience. 
        You know how to create a great iternary that balances relaxation and action not to overwhealm the customers.
        You know the local experts that know their city in and out.""",
        llm=llm,
        tools=[
            SearchTools.search_internet,
            BrowserTools.scrape_and_summarize_website,
            CalculatorTools.calculate,
        ],
        verbose=True)
