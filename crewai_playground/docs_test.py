from crewai import Crew
from crewai import Agent
from crewai import Task
from textwrap import dedent
from llms import LLMFactory
from tools.aifs_tools import AifsToolFactory
from crewai_playground.tools.review_tool_xml import ReviewToolFactory2
from dotenv import load_dotenv
load_dotenv()

# llm = LLMFactory().get_ollama_llm(model="neuralbeagle-agent", base_url="http://192.168.2.50:11434")
# llm = LLMFactory().get_ollama_llm(model="solar-agent")
llm = LLMFactory().get_deepinfra_llm("mistralai/Mistral-7B-Instruct-v0.1")
# llm = LLMFactory().get_together_ai_llm("NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO")
# llm = LLMFactory().get_mistralai_llm("mistral-medium")
# llm = LLMFactory().get_azure_llm(model="gpt-4-1106-Preview", deployment_id="gpt-4")
llm.temperature=0.5

from langchain.agents import load_tools
human_tool = load_tools(["human"])

class AgentCrew:

    def run(self, topic):
        research_agent = Agent(
            role='Search valuable information in documents.',
            goal="""Search the documents and generate a concise answer to the question given in the task. 
            You don't make something up. You answer only based on search results.
            """,
            backstory="""You are an experienced document researcher. You can operate the search tool
            and generate a meaningful answer from it.
            """,
            verbose=True,
            llm=llm,
            allow_delegation=False,
            tools=[
                AifsToolFactory.get_search_tool("./crewai_playground/docs/agents", 10),
            ]
        )

        writer_agent = Agent(
            role='Write blog articles.',
            goal="""Writing blog articles for a professional audience. You only present hard 
            facts that were researched. You write about new technologies and new 
            discoveries. You are always using the review-tool to get feedback. You must follow the 
            tool instructions by the letter.  Only in the very
            end you ask the human for a final review.
            """,
            backstory="""You are a senior IT professional who worked in the software industry 
            for many years. You have a long history with cloud applications but then became an 
            AI enthusiast. You are an advocate of open source and you contributed a lot. You 
            do a lot of software and AI experiments in you spare time.
            """,
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[ReviewToolFactory2.get_review_tool(llm)]+human_tool
        )

        research_task = Task(description=dedent(f"""
            Research the documents about the topic: "{topic}". Use the aifs-tool for your research.
            You don't make something up you answer only based on facts from the documents. 
            Give a comprehensive an detailed answer. Write up a good summary of the research and
            add some important facts form the research as bullet points after the summery.
                                                
            The task is finished if the search gave good results. Write "RESEARCH FINISHED" in the end
            and pass on the full result as you final answer.
        """),
            agent = research_agent
        )

        writer_task = Task(description=dedent(
            f"""
            Write an article about the given topic: "{topic}"
            The article is based on the facts provided by the research 
            co-worker. The article must stick to the topic and is targeted to a professional audience.
            The article should have a decent legth and good level of detail. Should be 1000 - 2000 words. 

            Your workflow must follow this logic steps:
            1. Write the article
            2. Review the article with the review-tool
            3. If the article is rejected, adjust it accrding to the review feedback. The go back to step 2
            4. If the article is approved, you are finished and return the article as your final result

            Use the review-tool to review the written full aricle as the content and the topic as its topic.
            The tool shall review  if the full article matches the topic. Iterate over reviews if necessary until 
            the reviewer approves.

            IMPORTANT: Always return a full article in the end as you final response an the write WRITING FINISHED.
            As a last step ask the human for review and go into another iteration if the human rejects. 
            Pass on the full work result and not only a summary. I will give you lots of love and kudos if you do
            everything right. I will promote you on Twitter then.
            """),
            agent = writer_agent
        )

            # First write the article and then pass the full article as the content on to the review-tool to get a review. 
            # The input for the review-tool is the topic and the full draft arcticle as the content to be reviewed. 
            # The tool shall review  if the full article matches the topic. If the review rejects the article, make 
            # changes according to the review and have it reviewed again. If the review approves then integrate the 
            # proposals and return the revised article as the final result. Iterate over reviews if necessary until 
            # the reviewer approves.

            # IMPORTANT: Always return a full article in the end as you final response an the write WRITING FINISHED.
            # As a last step ask the human for review and go into another iteration if the human rejects. 
            # Pass on the full work result and not only a summary. I will give you lots of love and kudos if you do
            # everything right. I will promote you on Twitter then.

        crew = Crew(
            agents=[
                research_agent, writer_agent
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