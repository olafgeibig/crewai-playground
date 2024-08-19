from crewai import Agent, Crew, Task, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, FileReadTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

def create_research_task(agent, resource, note_file) -> Task:
    return Task(
        description=f"""Research the most relevant websites about the resource "{resource}".
        Search topics: official, discussion, articles, howto, learning

        Perform the web search:
        - Search the web about for each of the topics
        - Each result must be related with the resource "{resource}". Don't add any general findings about the objectives.
        - Look at the URL and the snippet to validate each result for quality, accuracy, and relevance to the project.
        - If there are findings not related to the resource, ignore them. 
        - Only use maximum best 5 findings per objective.
        
        Generate the research result in markdown format.
        """,
        agent=agent,
        expected_output="research result"
    )

def create_refinement_task(agent, resource, note_file, input_task) -> Task:
    return Task(
        description=f"""Read the refinement section of the file "{note_file}" and extract the refinement objectives and the refinement template. Ignore the other sections.

        Iterate over all websites of the research result, for each website:
        - Read the content
        - Decide if it matches one of the refinement criterias. Match against the examples in the refinement criterias
        - Skip it if it doesn't match, if it does: Create a summary of the content with maximum 100 words and put it in the correct section of the refinement template 

        Generate the refinement result in markdown format according to the refinement template.
        """,
        agent=agent,
        expected_output="refinement result",
        context=[input_task]
    )

def create_writer_task(agent, resource, note_file) -> Task:
    return Task(
        description=f"""
        Write a note in markdown format about the resource "{resource}" using the note template.
        The note shall contain a general section explaining the resource and its concepts and a multiple themed section with important links to websites.
        
        Read the writing section of the file "{note_file}" and extract the note template and the note tags. Ignore the other sections.
        Read the file "refinement_result.md" and use it to write the note
        
        Description and concept:
        Scrape the content of the websites from official resources section using the website scraper tool and use the content for the following tasks:
        - Write a comprehensive description for the resource covering all aspects using 200 words min.
        - Explain the concepts used by the resource 200 words min.
        - Choose matching tags from the "note tags" and add them to the note.
        - Extract relevant links from the content of the official resources and add the links to the official resources section.

        The resources
        - Use the websites from the refinement result. Don't do your own research.
        - Copy the resources from the refinement result

        Generate the note in markdown format according to the note template, remove the '''markdown tag..
        """,
        agent=agent,
        expected_output="a note that is the filled template",
        output_file="note.md",
    )


def main():
    load_dotenv()
    # llm=ChatOpenAI(model="gpt-4o", temperature=0.7)
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    llm = ChatOpenAI(
        model="deepseek-chat", 
        api_key=DEEPSEEK_API_KEY, 
        base_url="https://api.deepseek.com/beta",
        temperature=0.0
    )

    research_agent = Agent(
        role='Research agent',
        goal='Research the the important resources for a given topic',
        backstory="I am a researcher with a great skill to find the most relevant resources. I stick to the objectives. I don't make something up",
        verbose=True,
        llm=llm,
        allow_delegation=False,
        tools=[FileReadTool(), SerperDevTool()]
    )

    refinement_agent = Agent(
        role='Refinement agent',
        goal='Validate and refine the findings of a website research so that it matches the given objectives. I use the scrape website tool to read the contents of a website',
        backstory='I am an analyst who thoroughly analyzes the findings of a research.',
        verbose=True,
        llm=llm,
        allow_delegation=False,
        tools=[FileReadTool(), ScrapeWebsiteTool()]
    )

    writer_agent = Agent(
        role='Note writer',
        goal='Write personal notes using a specified template.',
        backstory='I am a note writer with a great skill to write personal notes.',
        verbose=True,
        llm=llm,
        allow_delegation=False,
        tools=[FileReadTool(), ScrapeWebsiteTool()]
    )

    resource = "crewAI"
    note_file = "project_note.md"

    research_task = create_research_task(research_agent, resource, note_file)
    refinement_task =create_refinement_task(refinement_agent, resource, note_file, research_task)
    writer_task = create_writer_task(writer_agent, resource, note_file)

    crew = Crew(
        agents=[research_agent, refinement_agent, writer_agent],
        tasks=[writer_task], #research_task, refinement_task],#, writer_task],
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
