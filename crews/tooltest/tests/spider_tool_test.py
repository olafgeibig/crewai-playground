from tooltest.tools.spider_tool import SpiderTool
from crewai import Agent, Task, Crew

# def test_spider_tool():
#     spider_tool = SpiderTool()
    
#     searcher = Agent(
#         role="Web Research Expert",
#         goal="Find related information from specific URL's",
#         backstory="An expert web researcher that uses the web extremely well",
#         tools=[spider_tool],
#         verbose=True,
#         cache=False
#     )
    
#     choose_between_scrape_crawl = Task(
#         description="Scrape the page of spider.cloud and return a summary of how fast it is",
#         expected_output="spider.cloud is a fast scraping and crawling tool",
#         agent=searcher
#     )

#     return_metadata = Task(
#         description="Scrape https://spider.cloud with a limit of 1 and enable metadata",
#         expected_output="Metadata and 10 word summary of spider.cloud",
#         agent=searcher
#     )

#     css_selector = Task(
#         description="Scrape one page of spider.cloud with the `body > div > main > section.grid.md\:grid-cols-2.gap-10.place-items-center.md\:max-w-screen-xl.mx-auto.pb-8.pt-20 > div:nth-child(1) > h1` CSS selector",
#         expected_output="The content of the element with the css selector body > div > main > section.grid.md\:grid-cols-2.gap-10.place-items-center.md\:max-w-screen-xl.mx-auto.pb-8.pt-20 > div:nth-child(1) > h1",
#         agent=searcher
#     )

#     crew = Crew(
#         agents=[searcher],
#         tasks=[
#             choose_between_scrape_crawl, 
#             return_metadata, 
#             css_selector
#         ],
#         verbose=True
#     )
    
#     crew.kickoff()

def test_spider_tool():
    spider_tool = SpiderTool()
    
    searcher = Agent(
        role='Spider Agent',
        goal='Crawl a website and extract the essential information',
        backstory='An expert web researcher that uses the web extremely well. You know how to use a spider to crawl a website.',
        tools=[spider_tool],
        verbose=True,
        cache=False
    )
    
    do_crawl = Task(
        description="""
        Use the spider to get the website's content and then extract the relevant information from it.

        Crawl 'https://python.langchain.com/v0.1/docs/integrations/tools/'

        with the following options:
        - limit: 10
        - depth: 2

        The goal is to create a documentation of the tools in markdown format. The documentation 
        must contain the following information for each tool:
        1. Name of the tool
        2. Description
        3. Arguments and their meaning
        4. Requirements, e.g. needed API keys, needed libraries to import, config files, etc. 
        5. Possible use cases for an AI agent for this tool
        """,
        expected_output="""A list of tools in markdown format as follows:
        # Tool Name
        ## Description
        Description of the tool
        ## Python
        - module name
        - class name
        ## Arguments
        Bullteted list of arguments and their meaning 
        ## Requirements
        Bulleted list of requirements, e.g. needed API keys, needed libraries to import, config files, etc.
        ## Use Cases
        Bulleted list of use cases for agents
        """,
        agent=searcher,
        output_file="langchain_tools.md"
    )

    crew = Crew(
        agents=[searcher],
        tasks=[
            do_crawl
        ],
        verbose=True
    )
    
    crew.kickoff()

def test_spider_basics():
    spider_tool = SpiderTool()
    result = spider_tool.run(        
        url="https://python.langchain.com/v0.1/docs/integrations/tools/",
        mode="crawl",
        params={
            "limit": 2,
            "depth": 1,
            "metadata": True
        }
    )
    print(result)

if __name__ == "__main__":
    test_spider_tool()
    # test_spider_basics()