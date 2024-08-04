from crewai import Agent, Crew, Task
from crewai_tools import SerperDevTool, WebsiteSearchTool
from langchain_openai import ChatOpenAI


def create_research_task(agent, topic):
    return Task(
        description=f"""Research the knowledge base for authors writing about the topic {topic}.
        Search the internet about the topic and add the contents of found websites to the
        knowledge base by feeding their URL into the website search tool.
        """,
        agent=agent,
        expected_output="a search result"
    )


def create_writer_task(agent, topic, template):
    return Task(
        description=f"""Write a note about {topic} based on the template .
        Follow the instructions in the template file how to fill the placeholders.
        To do that you need to research the knowledge base with the website search tool.
        Temple: {template}
        """,
        agent=agent,
        expected_output="a note that is the fillied template",
        output_file="note.md"
    )


def main():
    research_agent = Agent(
        role='Research agent',
        goal='Research the the important resources for a given topic',
        backstory='I am a researcher with a great skill to find the most relevant resources.',
        verbose=True,
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.7),
        tools=[SerperDevTool(), WebsiteSearchTool()]
    )

    writer_agent = Agent(
        role='Note writer',
        goal='Write personal notes using a specified template.',
        backstory='I am a note writer with a great skill to write personal notes.',
        verbose=True,
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.7),
        tools=[WebsiteSearchTool()]
    )

    topic = "Crawl4AI project"
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
        {bulleted list of links to related know-how, blog posts, articles abpout the resource}
    """

    research_task = create_research_task(research_agent, topic)
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
