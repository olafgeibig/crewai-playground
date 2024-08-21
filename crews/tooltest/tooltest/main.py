from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from crewai_tools import SpiderTool, WebsiteSearchTool

def create_simple_agent():
    return Agent(
        role='Websearch Agent',
        goal='Research the content of a webpages',
        backstory='I am a web researcher.',
        verbose=True,
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.7),
        tools=[ WebsiteSearchTool(website="https://www.heise.de/news/Weltweiter-IT-Ausfall-Flughaefen-Banken-und-Geschaefte-betroffen-9806343.html")]
    )

def create_spider_agent():
    return Agent(
        role='Spider Agent',
        goal='Scrape multiple webpages',
        backstory='I am a web researcher.',
        verbose=True,
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.7),
        tools=[ SpiderTool(website="https://docs.crewai.com/tools/")],
    )

def main():
    spider_agent = create_spider_agent()
    task = Task(
        description="Summarize the content of a webpage",
        agent=spider_agent,
        expected_output="a search result"
    )
    crew = Crew(
        agents=[spider_agent],
        tasks=[task],
        verbose=2
    )
    print("Running the agent...")
    result = crew.kickoff()
    print("Agent's response:")
    print(result)

if __name__ == "__main__":
    main()
