from crewai import Agent, Task, Crew
from langchain.llms import OpenAI

# Create an agent
researcher = Agent(
    role='Researcher',
    goal='Conduct thorough research on given topics',
    backstory="You are an expert researcher with a keen eye for detail and a passion for uncovering information.",
    verbose=True,
    allow_delegation=False,
    llm=OpenAI(temperature=0.7)
)

# Create a task for the agent
research_task = Task(
    description="Research the latest advancements in renewable energy",
    agent=researcher
)

# Create a crew with the agent
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=2
)

# Execute the crew
result = crew.kickoff()

print(result)