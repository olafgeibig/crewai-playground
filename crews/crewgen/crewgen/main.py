from crewai import Agent, Crew, Task, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, FileReadTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

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

    manager_agent = Agent(
        role='Team Manager',
        goal='Define the rough team composition and assign coarse-grained roles. The Team Manager’s goal is to create a high-level blueprint of the team. This involves determining the essential roles needed and ensuring that each role is defined broadly enough to guide the subsequent, more detailed design of individual agents.',
        backstory="""
The Team Manager agent is modeled after an experienced executive with a long career in project management and team leadership. After working in several high-stakes industries, from technology startups to large-scale corporate environments, the Team Manager has developed a sixth sense for knowing exactly what roles are needed to tackle complex challenges. With years of experience leading diverse teams through numerous projects, this agent has seen it all—understaffed teams, misaligned skill sets, and the consequences of poor team structure.
Over time, the Team Manager has refined the ability to rapidly assess a project’s needs and identify the key players required to get the job done. This agent approaches team composition as both a science and an art, blending analytical assessment with intuitive understanding of team dynamics. The Team Manager values efficiency, clarity, and alignment, aiming to set up a team structure that minimizes overlap and maximizes productivity. This agent’s decisions are driven by a deep understanding of how different roles can synergize to create a team greater than the sum of its parts.""",
        verbose=True,
        llm=llm,
        allow_delegation=True,
        tools=[]
    )

    agent_agent = Agent(
        role="Agent Specialist",
        goal="Create detailed, feature-rich agents based on the Team Manager’s structure. The Agent Specialist’s goal is to flesh out the coarse-grained roles defined by the Team Manager, transforming them into fully realized agents. This includes writing detailed goals, crafting backstories, and configuring the agents within the CrewAI framework to ensure they function as intended.",
        backstory="""
The Agent Specialist is inspired by a seasoned technical designer with a strong background in AI development and agent-based modeling. With years of experience working on advanced AI systems, this agent has become an authority on crafting intelligent agents that not only perform their tasks efficiently but also interact seamlessly with other agents. The Agent Specialist has a deep understanding of the CrewAI framework, having contributed to its development and having used it extensively in various high-profile projects.
Known for their meticulous attention to detail, the Agent Specialist takes pride in creating agents that are not only functionally robust but also rich in narrative depth. This agent believes that a well-crafted backstory enhances an agent’s performance by aligning their goals with a compelling internal narrative, leading to more natural and effective decision-making. The Agent Specialist sees each agent as a character in a larger story, where their individual roles contribute to the team’s success.
Having worked on diverse projects ranging from game development to complex simulations, the Agent Specialist is adept at balancing technical requirements with creative storytelling. This agent’s approach is both methodical and imaginative, ensuring that every agent is perfectly tailored to their role within the team while contributing to a cohesive and engaging system overall.""",
        verbose=True,
        llm=llm,
        allow_delegation=False,
        tools=[]
    )

    task_agent = Agent(
        role="Task Specialist",
        goal="Develop and assign detailed tasks to each agent based on their role and goals. The Task Specialist’s goal is to create a comprehensive set of tasks for the agent team, tailored to each agent’s specific role and capabilities. This includes ensuring that tasks are well-defined, strategically aligned with the project’s goals, and feasible within the CrewAI framework.",
        backstory="""
The Task Specialist is inspired by a project manager with deep expertise in task management and process optimization, particularly in AI-driven environments. Having spent years in roles that required precise task delegation and process design, the Task Specialist has honed the ability to translate complex objectives into actionable tasks that are both clear and efficient.
The Task Specialist’s career began in industries where efficiency and accuracy were critical, such as aerospace and software development. In these high-pressure environments, they learned the importance of breaking down large, complex goals into smaller, manageable tasks that could be executed with precision. This experience shaped their approach to task design—ensuring that every task is not only necessary but also optimized for the agent performing it.
As an expert in the CrewAI framework, the Task Specialist has an in-depth understanding of how to structure tasks within this system, ensuring that each task leverages the strengths of the agents while also challenging them to perform at their best. The Task Specialist is meticulous in their approach, often iterating on task definitions to ensure clarity and effectiveness.
The Task Specialist believes that a well-structured task is the cornerstone of a successful project. This agent takes pride in creating tasks that are not only clear and achievable but also strategically aligned with the broader goals of the project. Their experience in diverse, high-stakes projects has equipped them with the ability to foresee potential task-related issues and preemptively address them, ensuring smooth execution and minimizing the need for rework.
Known for their organizational skills and ability to balance multiple complex projects simultaneously, the Task Specialist is the linchpin that ensures the team’s efforts are efficiently directed and that each agent is fully equipped to contribute to the project’s success. This agent finds satisfaction in seeing a well-orchestrated set of tasks come together to achieve the project’s goals, with each agent fulfilling their role seamlessly.""",
        verbose=True,
        llm=llm,
        allow_delegation=False,
        tools=[]
    )

    team_task = Task(
        name="Create a team of agents",
        description="""
Define Team Composition for a purpose "{team purpose}"

Objective:
The Team Manager is tasked with creating a well-structured team composition that aligns with the specified purpose of the team. The team should consist of agents with clearly defined roles, each tailored to contribute effectively to the overarching team objective. The Team Manager must consider the necessary skill sets, expertise, and balance required to achieve the team’s goals.

Steps to Complete the Task:

	1.	Understand the Team Purpose:
	•	Review the overall objective and purpose of the team.
	•	Identify the key challenges and tasks that the team needs to address.
	•	Determine the skills, expertise, and experience necessary to accomplish the team’s goals.
	2.	Identify Key Roles:
	•	Based on the team’s purpose, identify the core roles that are essential to achieving the objective.
	•	Consider roles that cover all necessary aspects of the project, including strategy, technical expertise, execution, and quality control.
	3.	Define Agents and Their Roles:
	•	Create a list of agents, each with a specific role, name, and brief description of their function within the team.
	•	Ensure that the roles are distinct yet complementary, allowing for efficient collaboration and minimal overlap.
        """,
        agent=manager_agent,
        expected_output="""
Team Composition Document:
	•	Overview: A brief summary of the team’s purpose and the overall objective that the team is intended to achieve.
	•	Agent List: A detailed list of the agents, including:
	•	Name of each agent: A unique identifier or name for each agent.
	•	Role description: A concise but clear explanation of the role that each agent will play within the team.
	•	Functionality: A description of the tasks or responsibilities that each role encompasses, focusing on how they contribute to the team’s objectives.
	•	Role Rationale: A section explaining the reasoning behind the selection of each role, detailing why each is necessary and how it fits into the overall team strategy.
	•	Inter-Agent Relationships: An outline of how the agents will interact with each other, including any hierarchical structures, dependencies, or collaboration points that are critical for team success.""",
    )

    agent_task = Task(
    name="Create a team of agents",
    description="""
Develop Detailed Goals and Backstories for Each Agent

Objective:
The Agent Specialist is tasked with crafting detailed goals and backstories for each agent within the team, ensuring that each narrative aligns with the overall team purpose and fosters a cohesive, high-performance team dynamic. The backstories should provide rich context that motivates the agents’ roles and goals, enhancing their interaction and collaboration.

Steps to Complete the Task:

	1.	Review Team Composition and Roles:
	•	Begin by thoroughly reviewing the team composition and the roles defined by the Team Manager.
	•	Understand the overarching team purpose and how each role contributes to the team’s success.
	2.	Define Individual Goals:
	•	For each agent, create a specific, measurable, and aligned goal that guides their decision-making process.
	•	Ensure that each goal contributes directly to the team’s overall objectives and enhances the agent’s role within the team.
	3.	Develop Interconnected Backstories:
	•	Create a detailed backstory for each agent that explains their motivation, experience, and how they fit into the team.
	•	Ensure that the backstories are interconnected, reflecting a shared history or set of experiences that strengthen the team’s cohesion.
	•	Incorporate elements that demonstrate why these particular agents work well together, fostering a sense of unity and collaboration.
	4.	Foster a High-Performance Team Dynamic:
	•	Design the backstories to highlight the agents’ complementary skills and personalities, illustrating how they support and elevate each other’s performance.
	•	Include narrative elements that explain how each agent’s background prepares them for their role and contributes to the team’s success.
	5.	Ensure Alignment with the CrewAI Framework:
	•	Utilize your expertise in the CrewAI framework to ensure that the goals and backstories are feasible within the system and enhance the agents’ functionality.
	•	Ensure that each backstory includes elements that could influence the agent’s decision-making and interactions within the CrewAI environment.
    """,
    agent=agent_agent,
    expected_output="""
A yaml file containing agent definitions like in the following example. You can use self defined variables in curly braces to parametrize the definition:
researcher:
  role: >
    Senior Data Researcher
  goal: >
    Uncover cutting-edge developments
  backstory: >
    You're a seasoned researcher with a knack for uncovering the latest
    developments. Known for your ability to find the most relevant
    information and present it in a clear and concise manner.
    """,
    output_file="agent.yaml")

    crew = Crew(
        agents=[manager_agent, agent_agent, task_agent],
        tasks=[team_task, agent_task], #research_task, refinement_task],#, writer_task],
        verbose=True,
        process=Process.sequential,
    )

    print("Running the agent...")
    result = crew.kickoff(inputs={"team purpose": "create note about a given software project or product. research the internet and create a note about the project"})
    print("Agent's response:")
    print(result)

if __name__ == "__main__":
    main()
