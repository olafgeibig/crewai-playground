from crewai import Crew
from crewai import Agent
from tools.browser_tools import BrowserTools
from tools.search_tools import SearchTools
from crewai import Task
from textwrap import dedent
from attic.utils.llms import LLMFactory
from dotenv import load_dotenv
load_dotenv()

llm = LLMFactory().get_ollama_llm("neuralbeagle-agent")
# llm = LLMFactory().get_anyscale_llm("mistralai/Mistral-7B-Instruct-v0.1")
llm.temperature = 0

llm_writer = LLMFactory().get_ollama_llm("neuralbeagle-agent")
llm_writer.temperature = 0.8

# llm = LLMFactory().get_together_ai_llm("NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO")

from langchain.agents import load_tools
human_tool = load_tools(["human"])

class BloggerCrew:
    # def __init__(self):


    def run(self, topic, resources, keywords, description):
        gather_agent = Agent(
            role='Gather valuable resources and add them to the knowledge base.',
            goal="""Extract the information from all the given resources to make them 
            available for co-workers. Obtain the resources and extract the content using the Browser tools. 
            In the end you store the cleaned content of the resources into the knowledge base.
            """,
            backstory="""You are an experienced knowledge extractor. You know how to recognise 
            the valuable information in a resource. You know how to maintain a knowledge base.
            """,
            verbose=True,
            llm=llm,
            allow_delegation=False,
            tools=[
                BrowserTools.scrape_and_summarize_website,
            ]
        )

        research_agent = Agent(
            role='Researches keywords on the internet and add the contents of relevant search results to the knowledge base.',
            goal="""Research given topics or keywords and identify good resources. When doing 
            internet searches, read the snippets of the search results and follow only links 
            that are related to the overall research goal. In the end you store the cleaned 
            content of the resources into the knowledge base. 
            
            Use the search tool to do internet searches. Use the browser tool to load and summarize the search results.
            """,
            backstory="""You are an experienced researcher, You can do internet searches like 
            a pro. You have a talent to immediately know which search results are relevant.
            """,
            verbose=True,
            llm=llm,
            allow_delegation=False,
            tools=[
                BrowserTools.scrape_and_summarize_website,
                SearchTools.search_internet
            ]
        )

        writer_agent = Agent(
            role='Write blog articles.',
            goal="""Writing blog articles for a professional audience. You only present hard 
            facts that were thoroughly researched. You write about new technologies and new 
            discoveries. You are utilising the knowledge base for delivering the facts for 
            your articles.
            """,
            backstory="""You are a senior IT professional who worked in the software industry 
            for many years. You have a long history with cloud applications but then became an 
            AI enthusiast. You are an advocate of open source and you contributed a lot. You 
            do a lot of software and AI experiments in you spare time.
            """,
            verbose=True,
            llm=llm_writer,
            tools=[
                # BrowserTools.scrape_and_summarize_website
            ]+human_tool
        )

        gather_task = Task(description=dedent(f"""
            Gather all the given resources needed for writing a blog article. You contribute to 
            this effort. Iterate over all of the resources and obtain, download or parse them and 
            extract the content. You must use the browser tool to load a URL resource from the internet and 
            extract the main content. Process the content and store it in the knowledge base. 

            Processing a resource and add it to the knowledge base means to summarize the resource 
            content. In the end the knowledge base is a markdown document with a summary for each 
            resource. The knowledge base must be passed on to the co-worker that works on the next task.

            The blog article:
            Topic: {topic}
            Resources: {resources}

            The task is finished when all resources have been collected into the knowledge base. 
            Write "GATHER FINISHED" in the end and hand over the knowledge base to the next co-worker.
        """),
            agent = gather_agent
        )

        research_task = Task(description=dedent(f"""
            Get the knowledge base from the previous co-worker and extend it with your research.
            Research all the given keywords in the context of the article effort. The research 
            contributes to the article effort by adding the research results to the knowledge base. 
            Use the article topic and the description as the relevance criteria for evaluation of 
            the research results. Iterate over all of the comma seperated keyword and do these steps:
                                                
            1. Do an internet search: [keyword] [article topic]
            2. Evaluate each search result by reading their snippet and source if they are relevant. Choose only the three most relevant search results
            3. Load the relevant search results with the browser tool and extract their contents
            4. Process each extracted content and process it
            5. Add the processed content to the exiting knowledge base

            Processing a content and add it to the knowledge base means to summarise the content. 
            In the end the knowledge base is a markdown document with a summary for each resource. 
            The previously existing knowledge base must be continued and extended with your results. 
            The knowledge base must be passed on to the co-worker that works on the next task.

            The article:
            Topic: {topic}
            Keywords: {keywords}
            Description: {description}

            The task is finished if all keywords had been researched and their processed contents 
            added to the knowledge base. Write "RESEARCH FINISHED" in the end and hand over the knowledge base 
            to the next co-worker.
        """),
            agent = research_agent
        )

        writer_task = Task(description=dedent(f"""
            Write the actual article based on the facts provided by the previously created knowledge 
            base that contains the resources and the results of the keyword research . The article must
            stick to the topic and is targeted to a professional audience. Use the given description 
            to understand the goal of the article. Let this guide your writing. Obey the topic and 
            comply to the description

            The article:
            Topic: {topic}
            Resources: {resources}
            Keywords: {keywords}
            Description: {description}

            When the article is completely written ask the human for a review Obey the human's change 
            requests and update the article accordingly. The task is finished when the human approves it.
        """),
            agent = writer_agent
        )

        crew = Crew(
            agents=[
                gather_agent, research_agent, writer_agent
            ],
            tasks=[gather_task, research_task, writer_task],
            verbose=True
        )
        result = crew.kickoff()
        return result

if __name__ == "__main__":
    print("## Welcome to Blogger Crew")
    print('-------------------------------')
    blogger_crew = BloggerCrew()
    result = blogger_crew.run(
        topic="CrewAI is a new and cool agent framework",
        resources="""
            https://github.com/joaomdmoura/CrewAI/wiki/Defining-Tasks, 
            https://github.com/joaomdmoura/CrewAI/wiki/Understanding-Agents, 
            https://github.com/joaomdmoura/CrewAI/wiki/Managing-Processes,
            https://github.com/joaomdmoura/CrewAI/wiki/Delegation-and-Collaboration, 
            https://github.com/joaomdmoura/CrewAI/wiki/Agent-Tools
        """,
        keywords="AI-agents, agentic software",
        description="""
        An article about the advantages of CrewAI. The final article must have 1000 words. Well written in a fluid form with paragraphs. 
        No bullet points and numbered lists. Always use the tools. The knowledge base must always be passed on between coworkers and new 
        knowledge is appended to the aexiting knowledge base."""
    )
    print(result)

        #    https://github.com/joaomdmoura/CrewAI/wiki/Defining-Tasks, 
        #     https://github.com/joaomdmoura/CrewAI/wiki/Managing-Processes,
        #     https://github.com/joaomdmoura/CrewAI/wiki/Delegation-and-Collaboration, 
        #     https://github.com/joaomdmoura/CrewAI/wiki/Agent-Tools

            # Use quotes from the provided resources where it makes sense and validates the article's point. 
            # The quotes can be used to add more details to central points of the article or provide background. 
            # If quotes are used, add a reference to the resource with a footnote. Use the browser tool to 
            # load a resource and extract it's content. Read the content and find maximum three important 
            # passages that you can quote. Align the quote with the topic and description of the article