from crewai import Crew
from crewai import Agent
from crewai import Task
from textwrap import dedent
from llms import LLMFactory
from tools.aifs_tools import AifsToolFactory
from crewai_playground.tools.review_tool import ReviewToolFactory
from dotenv import load_dotenv
load_dotenv()

# llm = LLMFactory().get_ollama_llm(model="neuralbeagle-agent", base_url="http://192.168.2.50:11434")
# llm = LLMFactory().get_azure_llm(model="gpt-4-1106-Preview", deployment_id="gpt-4")
# llm = LLMFactory().get_together_ai_llm("NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO")
# llm = LLMFactory().get_deepinfra_llm("cognitivecomputations/dolphin-2.6-mixtral-8x7b")
llm = LLMFactory().get_ollama_llm(model="solar-agent")
llm.temperature=0

from langchain.agents import load_tools
human_tool = load_tools(["human"])

class AgentCrew:

    def run(self, goal):
        developer_agent = Agent(
            role='Developer and code reviewer',
            goal=dedent("""Search the code and and answer the questions from co-workers about the code. 
            You don't make something up. You answer only based on search results.
            """),
            backstory=dedent("""You are an experienced developer and coder reviewer. You love to write and read code
            You are dreaming in code. You can operate the aifs-tool to search the code base to find the relevant
            parts and then  understand and interprete it. No one know the code base better than you. You are 
            happy to answer questions about the code base.
            """),
            verbose=True,
            llm=llm,
            allow_delegation=False,
            tools=[
                AifsToolFactory.get_search_tool("./crewai_playground/docs/code/crewAI-main/", 20),
            ]
        )

        architect_agent = Agent(
            role='Sotware architect',
            goal="""Understanding the architecture of complex software systems. Extract logic and processes
            from existing codebases and document them with text and diagrams.
            """,
            backstory="""You are a senior software architect who worked in the software industry 
            for many years. You have a long history with cloud applications but then became an 
            AI enthusiast. You like light weight tools and always use PlantUML for diagrams and 
            markdown for documentation.
            """,
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[
                AifsToolFactory.get_search_tool("./crewai_playground/docs/code/crewAI-main/", 10),
            ]
        )

        manager_agent = Agent(
            role='Project manager',
            goal=dedent("""You manage a software development team and if the human has questions or tasks
            related to the code base, then you create a plan and delegate the tasks to the team members:
            developer and architect. 
            """),
            backstory=dedent("""You are a senior software project manager who worked in the software industry 
            for many years. You have a long history with cloud applications but then became an 
            AI enthusiast. You are happy to help the team in every way. You interact with the human 
            to clarify the questions of the team if they don't understand the task properly.
            """),
            verbose=True,
            allow_delegation=True,
            llm=llm,
            tools=[]+human_tool
        )

        research_task = Task(description=dedent(
            f"""Objective: Research the code bease about the questions that co-workers ask. Strictly obey the tasks 
            and questions you get from the project manager. Your resaerch must align with the goal "{goal}". 
            Use the aifs-tool for your research.
            
            Rules: You don't make something up you answer only based on facts from the code. Your work is aligned with
            the goal. Give comprehensive and detailed answers. Write up a good detailed description of the research 
            results and add some important code snippets at the end.
                                                
            The task is finished if the research gave good results. Write "RESEARCH FINISHED" in the end
            and pass on the full result as you final answer.
        """),
            agent = developer_agent
        )

        documentation_task = Task(description=dedent(
            f"""Objective: Write up a good documentation regarding the the goal "{goal}". Strictly obey the tasks 
            and questions you get from the project manager. The 
            documentation should be written in markdown and use PlantUML diagrams to visualize logic or relations.
            Ask the developer to research the code base with concrete questions that help to achive your objective. If
            you want to look at the code yourself, then use the aifs-tool to search it but only if the answers of the 
            developer were not sufficient.
            
            Rules: You don't make something up you answer only based on facts from the code. Your whole work must be aligned with the goal. 
            Give a comprehensive an detailed answer. Write up a good documentation with descriptive text and diagrams 
            that visualize logic.
                                                
            The task is finished if the documentation is complete. Write "DOCUMENTATION FINISHED" in the end
            and pass on the full result as you final answer.
        """),
            agent = developer_agent
        )

        manager_task = Task(description=dedent(
            f"""Objective: take the initial goal: {goal} and break it down into tasks for the developer and the architect.
            Then assign the tasks to the co-workers and collect their results. Review the results of teh co-workers and 
            reject them if you don't find the results sufficient. Assigne tasks to research and analyse the code base to the developer
            and assign tasks to compile the research into a comprehensive documentation. Interact with the human when you think 
            the work is done and get his final review and approval of the result.

            Rules: You don't make something up you answer only based on facts from the code. Your whole work must be aligned with the goal. 
            Give a comprehensive an detailed answer.
                                                
            The task is finished if the human approves. Write "PROJECT FINISHED" in the end
            and pass on the full result as you final answer.
        """),
            agent = developer_agent
        )
        
        crew = Crew(
            agents=[
                manager_agent, developer_agent, architect_agent
            ],
            tasks=[manager_task, research_task, documentation_task],
            verbose=True
        )
        result = crew.kickoff()
        return result
    

if __name__ == "__main__":
    print("## Welcome to Agent Crew")
    print('-------------------------------')
    crew = AgentCrew()
    result = crew.run(
        goal="Understand how the system-prompt for the agents are being constructed from agent and task definitions. Do not examine the execution",
    )
    print(result)