from langchain.agents import load_tools
from textwrap import dedent

from crewai import Agent, Crew, Task
from crewai_tools.tools.rag.rag_tool import RagTool
from dotenv import load_dotenv
from tools.aifs_tools import AifsToolFactory
from tools.review_tool import ReviewToolFactory

from attic.utils.llms import LLMFactory

load_dotenv()

# llm = LLMFactory().get_ollama_llm(model="neuralbeagle-agent", base_url="http://192.168.2.50:11434")
# llm = LLMFactory().get_ollama_llm(model="solar-agent")
# llm = LLMFactory().get_deepinfra_llm("mistralai/Mistral-7B-Instruct-v0.1")
llm = LLMFactory().get_together_ai_llm(
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO")
# llm = LLMFactory().get_mistralai_llm("mistral-medium")
# llm = LLMFactory().get_azure_llm(model="gpt-4-1106-Preview", deployment_id="gpt-4")
# llm = LLMFactory().get_ollama_llm(model="dolphin-2.6-mistral-dpo-laser:7B-Q5_K_M")
# llm = LLMFactory().get_ollama_llm(model="openhermes-agent")
# llm = LLMFactory().get_deepinfra_llm("cognitivecomputations/dolphin-2.6-mixtral-8x7b")
# llm = LLMFactory().get_anyscale_llm("mistralai/Mistral-7B-Instruct-v0.1")
# llm = LLMFactory().get_together_ai_llm("teknium/OpenHermes-2p5-Mistral-7B")
# llm = LLMFactory().get_ollama_llm(model="memgpt-agent:7B-Q5_K_M")
# llm = LLMFactory().get_openai_llm(model="gpt-4-1106-preview")
llm.temperature = 0.7
# llm.num_ctx=4096


human_tool = load_tools(["human"])


class AgentCrew:

    def run(self, topic):
        research_agent = Agent(
            role='a researcher for valuable information in documents',
            goal=dedent("""\
Extracting the information that is matching the given research goal. Returning great insights \
that enables co-workers to gain the knowledge they need to do their work. Always deliver truthful \
information that is solely based on facts from the documents.
                        """),
            backstory=dedent("""\
You are an experienced document researcher who knows how to do a semantic search with the "aifs-tool" to \
extract the requested information from the documents. You never make somtheing up. Your results \
are always solely based on the researched findings.
                            """),
            verbose=True,
            llm=llm,
            allow_delegation=False,
            tools=[
                # AifsToolFactory.get_search_tool("./crewai_playground/docs/agents", 30),
                RagTool().from_directory("./crewai_playground/docs/agents")
            ]
        )

        writer_agent = Agent(
            role='blog article writer',
            goal=dedent("""\
Writing blog articles for a professional audience, only using hard \
facts that were researched. Writing about new technologies and new \
discoveries. Wring a fresh and clear style in paragraphs without boring lists of points.
            """),
            backstory=dedent("""\
You are a senior IT professional who worked in the software industry \
for many years. You have a long history with cloud applications but then became an \
AI enthusiast. You are an advocate of open source and you contributed a lot. You \
do a lot of software and AI experiments in you spare time.
            """),
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[]+human_tool
        )

        research_task = Task(description=dedent(f"""\
Research the documents about the topic: {topic}. Workflow:
1. Search the documents
2. Write up a comprehensive and long summary of the search results.
3. Add a list with the most important quotes from the search results

The task is finished if the search gave good results. Write "RESEARCH FINISHED" in the end \
and pass on your complete result as decribes above as you final answer.
        """),
            expected_output="The research result",
            agent=research_agent
        )

        writer_task = Task(description=dedent(f"""\
Write an article about the given topic: "{topic}" \
The article is based on the facts provided by the research \
co-worker. The article must stick to the topic and is targeted to a professional audience. \
The article should have a decent legth and good level of detail. Should be 1000 - 2000 words.

Your workflow must follow this logic steps:
1. Write the article
2. Ask the human to review your article
3. If the article is rejected, you must adjust it according to the review feedback. Then go back to step 2
4. If the article is approved, you are finished and return the article as your final result
            """),
            # Use the "review-tool" to get a review of your full aricle. Stricly follow the instructions \
            # in the tool's description how to call it. \
            # The tool shall review if the full article matches the topic. Iterate over crycles of review and revision \
            # if necessary until the reviewer approves.

            expected_output="The full final article",
            context=[research_task],
            agent=writer_agent
        )

        # IMPORTANT: Always return a full article in the end as you final response an the write WRITING FINISHED.
        # As a last step ask the human for review and go into another iteration if the human rejects.
        # Pass on the full work result and not only a summary. I will give you lots of love and kudos if you do
        # everything right. I will promote you on Twitter then.

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
