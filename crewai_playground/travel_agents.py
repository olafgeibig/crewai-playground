from crewai import Agent
import os
from tools.browser_tools import BrowserTools
from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools
from tools.sec_tools import SECTools

from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool

from langchain.chat_models import ChatOpenAI
from langchain.llms import Ollama

from dotenv import load_dotenv
load_dotenv()

# llm = ChatOpenAI(model='gpt-3.5-turbo-1106') # Loading GPT-3.5
# llm = ChatOpenAI(
#     # model="mistralai/Mixtral-8x7B-Instruct-v0.1",
#     model="mistralai/Mistral-7B-Instruct-v0.1",
#     api_key=os.getenv('ANYSCALE_API_KEY'),
#     base_url=os.getenv('ANYSCALE_BASE_URL')
# )
# llm = Ollama(model="dolphin-agent")
llm = Ollama(model="nous-hermes-2-solar")

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
        role='Local Expert at this city',
        goal="""Provide the BEST insights about the selected city. Check the security information for the destination.
        Find all the secret tipps and amazing experiences off the beaten path.
        """,
        backstory="""A knowledgeable local guide with extensive information
        about the city, it's attractions and customs. You know all the secrets that most tourists don't know.
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
