from crewai import Agent, Crew, Task
from crewai_tools import SerperDevTool, WebsiteSearchTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

def create_research_task(agent, topic, research):
    return Task(
        description=f"""Research the most relevant web resources about the topic "{topic}".
        Search the web about the topic for each of these categories:
        {research}
        Each result must have to do with the topic "{topic}". Don't add any general findings.
        Only use maximum best 5 findings per category.
        The result is a list of the top relevant links to resources for each of the category.
        """,
        agent=agent,
        expected_output="research result"
    )


def create_writer_task(agent, topic, template):
    return Task(
        description=f"""Write a note about {topic} based on the template .
        Follow the instructions in the template file how to fill the placeholders.
        Research each link from the research result with the website search tool.
        Use the official resources for the description an concept.
        Use the findings to fill the bulleted lists.
        Don't do your own research.
        Temple: {template}
        """,
        agent=agent,
        expected_output="a note that is the filled template",
        output_file="note.md"
    )


def main():
    load_dotenv()
    research_agent = Agent(
        role='Research agent',
        goal='Research the the important resources for a given topic',
        backstory='I am a researcher with a great skill to find the most relevant resources.',
        verbose=True,
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.7),
        allow_delegation=False,
        tools=[SerperDevTool()]
    )

    writer_agent = Agent(
        role='Note writer',
        goal='Write personal notes using a specified template.',
        backstory='I am a note writer with a great skill to write personal notes.',
        verbose=True,
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.7),
        allow_delegation=False,
        tools=[WebsiteSearchTool()]
    )

    topic = "Crawl4AI project"
    research = """1. official resources like source code, website, research paper
    2. community resources like discussion forum, discord, slack, X account
    3. Know-How resources like articles, blog-posts
    """
    template = """
        ---
        created: {date}
        updated: {date}
        type: resource
        tags: {One or more of: ai/dev, ai/agents, ai/tools, ai/science}
        description: {short description of the resource}
        ---
        # Description
        {long description of the resource}
        # Resources
        ## Official
        {bulleted list of links for the resource pointing to official resources, github link, website}
        ## Community
        {bulleted list of links to community resources}
        # Know-How
        {bulleted list of links to related know-how, blog posts, articles about the resource}
    """

    research_task = create_research_task(research_agent, topic, research)
    writer_task = create_writer_task(writer_agent, topic, template)

    crew = Crew(
        agents=[research_agent, writer_agent],
        tasks=[research_task, writer_task],
        verbose=2
    )
    print("Running the agent...")
    result = crew.kickoff()
    print("Agent's response:")
    print(result)


if __name__ == "__main__":
    main()
