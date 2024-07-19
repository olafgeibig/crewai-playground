from crewai import Agent, Task, Crew
from langchain.llms import OpenAI

def create_simple_agent():
    return Agent(
        role='Simple Agent',
        goal='Perform simple tasks and respond to queries',
        backstory='I am a helpful AI assistant created to assist with various tasks.',
        verbose=True,
        llm=OpenAI(temperature=0.7)
    )

def main():
    simple_agent = create_simple_agent()
    task = Task(
        description="Say hello to the world",
        agent=simple_agent
    )
    crew = Crew(
        agents=[simple_agent],
        tasks=[task],
        verbose=2
    )
    print("Running the agent...")
    result = crew.kickoff()
    print("Agent's response:")
    print(result)

if __name__ == "__main__":
    main()
