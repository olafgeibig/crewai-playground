from crewai import Crew
from crewai import Agent
from crewai import Task
from textwrap import dedent
from llms import LLMFactory
from tools.aifs_tools import AifsToolFactory
from tools.reviewer_tool import ReviewerToolFactory
from dotenv import load_dotenv
load_dotenv()

llm = LLMFactory().get_ollama_llm(model="neuralbeagle-agent", base_url="http://192.168.2.50:11434")
# llm = LLMFactory().get_together_ai_llm("NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO")

from langchain.agents import load_tools
human_tool = load_tools(["human"])

class AgentCrew:

    def run(self, topic):
        rag_agent = Agent(
            role='Search valuable information in documents.',
            goal="""Search the documents and generate a concise answer to the question given in the task. You don't make something up. You answer only based on search results.
            """,
            backstory="""You are an experienced document researcher. You can operate the search tool
            and genrate a meaningful answer from it.
            """,
            verbose=True,
            llm=llm,
            allow_delegation=False,
            tools=[
                AifsToolFactory.get_search_tool("/Users/olaf/work/ai/crewai-playground/docs/", 10),
            ]
        )

        writer_agent = Agent(
            role='Write blog articles.',
            goal="""Writing blog articles for a professional audience. You only present hard 
            facts that were thoroughly researched. You write about new technologies and new 
            discoveries. You are using the reviewer_tool to get feedback. 
            """,
            backstory="""You are a senior IT professional who worked in the software industry 
            for many years. You have a long history with cloud applications but then became an 
            AI enthusiast. You are an advocate of open source and you contributed a lot. You 
            do a lot of software and AI experiments in you spare time.
            """,
            verbose=True,
            allow_delegation=True,
            llm=llm,
            tools=[ReviewerToolFactory.get_reviewer_tool(llm)]+human_tool
        )

        research_task = Task(description=dedent(f"""
            Research the documents about the topic: "{topic}". Use the search tool for your research.
            You don't make something up you answer only based on facts from the documents. 
            Summarize the search result and give a comprehensive an detailed answer. If the search did not
            give enough good results, then search again. 
            Pass on your research results to the writer coworker.
                                                
            The task is finished if the search gave good results, otherwise search again.
            added to the knowledge base. Write "RESEARCH FINISHED" in the end and hand over the knowledge base 
            to the next co-worker.
        """),
            agent = rag_agent
        )

        writer_task = Task(description=dedent(
            f"""
            Write the actual article based on the facts provided by the previously created knowledge 
            base that contains the resources and the results of the keyword research . The article must
            stick to the topic and is targeted to a professional audience. Use the given description 
            to understand the goal of the article. Let this guide your writing. Obey the topic and 
            comply to the description.
            Use the reviewer_tool to get a review. Make changes according to the review and if the
            reviewer approves, then return the revised article as the final result and the task is finished.

            The article topic: {topic}
        """),
            agent = writer_agent
        )
            # When the article is completely written ask the human for a review Obey the human's change 
            # requests and update the article accordingly. The task is finished when the human approves it.
        crew = Crew(
            agents=[
                rag_agent, writer_agent
            ],
            tasks=[research_task, writer_task],
            verbose=True
        )
        result = crew.kickoff()
        return result
    

if __name__ == "__main__":
    print("## Welcome to Agent Crew")
    print('-------------------------------')
    blogger_crew = AgentCrew()
    result = blogger_crew.run(
        topic="What are the capabilities of Personal Agents",

    )
    print(result)