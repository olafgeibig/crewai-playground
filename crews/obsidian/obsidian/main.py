from crewai import Agent, Crew, Task, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

def create_research_task(agent, resource, objectives) -> Task:
    return Task(
        description=f"""Research the most relevant websites about the resource "{resource}".
        Search the web about the resource for each of these categories:
        {objectives}
        Each result must be related with the resource "{resource}". Don't add any general findings about the objectives.
        If there are no findings related to the resource, ignore them. 
        Only use maximum best 5 findings per category.
        The result is a list of the top relevant links to websites for each of the category in markdown format:

        # Resources
        ## Official
        ## Community
        ## Know-How
        """,
        agent=agent,
        expected_output="research result"
    )

def create_validation_task(agent, resource, objectives, input_task) -> Task:
    return Task(
        description=f"""Validate the research result for websites about the resource "{resource} by checking the following objectives:
        {objectives}
        If findings are not related to the topic or do not match the objectives, reject the research result and give your reasons.
        In case of rejection, the researcher must improve the reaserch.
        """,
        agent=agent,
        expected_output="validation result",
        context=[input_task]
    )

def create_writer_task(agent, resource, template, input_task) -> Task:
    return Task(
        description=f"""Write a note in markdown format about the resource "{resource}" using the template.
        The note shall contain a general section explaining the resource and its concepts and multiple themed section with important links to websites.
        Replace the placeholders in the template as follows:
        
        Description and concept:
        - Iterate over all links of teh official resources. Iterate over all findings and examine the content.
        - Write a concise description of the resource.
        - Extract the concept used by the resource.
        - Use the official resources for the description an concept.
        - Use the know-how resources for the description and concept.
        - Follow relevant links from the content.

        The resource lists
        - Use the websites from the resources section of the research result. Don't do your own research.
        - Create a bullet point for each resource finding in the format: markdown-link caption
        - Take care that only links matching the section are placed with a section. Resort if necessary.

        Template: {template}
        """,
        agent=agent,
        expected_output="a note that is the filled template",
        output_file="note.md",
        context=[input_task]
    )


def main():
    load_dotenv()
    research_agent = Agent(
        role='Research agent',
        goal='Research the the important resources for a given topic',
        backstory="I am a researcher with a great skill to find the most relevant resources. I stick to the objectives. I don't make something up",
        verbose=True,
        llm=ChatOpenAI(model="gpt-4o", temperature=0.7),
        allow_delegation=False,
        tools=[SerperDevTool()]
    )

    validator_agent = Agent(
        role='Validator agent',
        goal='Validate if the findings of a research are valid against the give research topic.',
        backstory='I am a validator that thoroughly validates the findings of a research.',
        verbose=True,
        llm=ChatOpenAI(model="gpt-4o", temperature=0.7),
        allow_delegation=True,
        tools=[]
    )

    writer_agent = Agent(
        role='Note writer',
        goal='Write personal notes using a specified template.',
        backstory='I am a note writer with a great skill to write personal notes.',
        verbose=True,
        llm=ChatOpenAI(model="gpt-4o", temperature=0.7),
        allow_delegation=False,
        tools=[WebsiteSearchTool()]
    )

    resource = "crewAI"
    research_objectives = """1. official resources like project website, source code, related research papers
    2. community resources like discussion forums, discord, slack, X account
    3. Know-How resources like articles, blog-posts, social media posts, videos, podcasts
    """
    template = """
        ---
        created: {today's date}
        updated: {today's date}
        type: resource
        tags: {One or more of: ai/dev, ai/agents, ai/tools, ai/science}
        description: {short description of the resource}
        ---
        # Description
        {long description of the resource}
        ## Concepts
        {Explanation of the concepts used by the resource}
        ## Usages
        {Explanation of the usages of the resource}
        # Resources
        ## Official
        {bulleted list of links to official resources}
        ## Community
        {bulleted list of links to community resources}
        ## Know-How
        {bulleted list of links to know-how websites}
    """

    research_task = create_research_task(research_agent, resource, research_objectives)
    validation_task =create_validation_task(validator_agent, resource, research_objectives, research_task)
    writer_task = create_writer_task(writer_agent, resource, template, validation_task)

    crew = Crew(
        agents=[research_agent, validator_agent, writer_agent],
        tasks=[research_task, validation_task, writer_task],
        verbose=True,
        process=Process.sequential,
        # planning=True,
        # planning_llm=ChatOpenAI(model="gpt-4o")
    )
    print("Running the agent...")
    result = crew.kickoff()
    print("Agent's response:")
    print(result)


if __name__ == "__main__":
    main()
