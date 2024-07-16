from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import ScrapeWebsiteTool
from langchain.agents import load_tools
from src.utils.llms import LLMFactory
from textwrap import dedent
import os
from dotenv import load_dotenv
load_dotenv()
# llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
# llm = LLMFactory().get_anyscale_llm("mistralai/Mixtral-8x7B-Instruct-v0.1")
# llm = LLMFactory().get_ollama_llm(model="olafgeibig/hermes-2-pro:Q5_K_M") #, base_url="http://192.168.2.50:11434")
# llm = LLMFactory().get_openai_llm(model="phi-2-openhermes-2.5:2.7B-Q5_K_M", base_url="http://localhost:11434/v1/", api_key="foo")
# llm = LLMFactory().get_azure_llm("gpt-4-1106-Preview", "gpt-4")
# llm = LLMFactory().get_openai_llm(model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key=os.getenv("DEEPINFRA_API_KEY"), base_url="https://api.deepinfra.com/v1/openai")
# llm = LLMFactory().get_google_ai_llm("gemini-pro")
# llm = LLMFactory().get_together_ai_llm("teknium/OpenHermes-2p5-Mistral-7B")
# llm = LLMFactory().get_openai_llm(model="mistralai/Mistral-7B-Instruct-v0.1", api_key=os.getenv("TOGETHERAI_API_KEY"), base_url=os.getenv("TOGETHERAI_BASE_URL"))
# llm = LLMFactory().get_anyscale_llm("mistralai/Mistral-7B-Instruct-v0.1")
smart_llm = LLMFactory().get_anthropic_llm("claude-3-haiku-20240307")
summarizer_llm = LLMFactory().get_together_ai_llm("teknium/OpenHermes-2p5-Mistral-7B")
# smart_llm = summarizer_llm
# llm.temperature = 0.1

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import BraveSearch
from langchain_community.tools.tavily_search import TavilySearchResults
# search_tool = BraveSearch.from_api_key(api_key=os.getenv('BRAVE_API_KEY'), search_kwargs={"count": 10})
search_tool = TavilySearchResults()
scrape_tool = ScrapeWebsiteTool()
# human_tool = load_tools(["human"])

# Define your agents with roles and goals
topic = "Impact of million-plus token context window language models on RAG"

researcher = Agent(
  role='Senior Researcher',
  goal="""Gather, analyze, and compile information on a given topic by conducting web 
  searches, identifying relevant resources, delegating research tasks to assistants, 
  and generating a comprehensive search report that includes references to the found 
  resources.""",
  backstory="""You are an experienced researcher with a strong background in conducting 
  research across various domains. Your expertise and decision-making skills are highly 
  valued by your research institution, and you are expected to lead and manage a team of 
  research assistants effectively.""",
  verbose=True,
  allow_delegation=True,
  tools=[search_tool],
  llm=smart_llm
)
resource_summarizer = Agent(
  role='Resource Summarizer',
  goal="""Support the Senior Researcher by reading and summarizing web resources 
  on a given topic. You will be assigned specific resources to read, understand, and 
  summarize, contributing to the creation of a comprehensive search report. You are
  using the scrape tool to extract the content from the webpages.""",
  backstory="""You are a skilled assistant with a keen eye for detail and the ability 
  to quickly grasp and synthesize complex information. Your role is crucial in assisting 
  the Senior Researcher in conducting efficient and effective research across various domains.""",
  verbose=True,
  allow_delegation=False,
  tools=[scrape_tool],
  llm=summarizer_llm
)

# writer = Agent(
#   role='Tech Content Strategist',
#   goal='Craft compelling content on tech advancements',
#   backstory="""You are a renowned Tech Content Strategist, known for your insightful
#   and engaging articles on technology and innovation. With a deep understanding of
#   the tech industry, you transform complex concepts into compelling narratives.""",
#   verbose=True,
#   allow_delegation=True,
#   llm=llm,
#   function_calling_llm=llm_func
# )


websearch_task = Task(
  description=dedent(f"""Web Search and Resource Identification. Topic: {topic}
  - Perform web searches using relevant keywords and phrases related to the given research topic.
  - Analyze the search results based on the URL, title, and snippet to determine their potential relevance and quality. Give the resources
    a relavance score 0-100.
  - Maintain a list of promising relevant resources for further investigation.
  - If the initial search results are unsatisfactory, modify the search query iteratively to improve the quality and relevance of the results.
  - After each search iteration, analyze the new results and update the list of promising relevant resources. Remove less relevant
    and add more relevant. The goal is a list with the best resources for the given topic.
  """),
  expected_output="A list of the 10 most relevant resources in JSON format: [{\"title\": \"\", \"url\": \"\", \"relevance\": \"\"},]",
  output_file='resource_list.md',
  agent=researcher
)

delegation_task = Task(
  description=dedent("""Delegation to Research Assistants.
  - Assign the identified resources to your team of research assistants for in-depth reading and summarization.
  - Provide clear instructions to the assistants regarding the specific aspects of the resource to focus on and the desired format of the summary.
  - Monitor the progress of the research assistants and provide guidance or reassign tasks as necessary.
  - Collect the summaries from the assistants"""),
  expected_output="A collection of summaries each with a reference to its source.",
  agent=researcher
)

report_task = Task(
  description=dedent(f"""Compilation of a Research Report. Topic: {topic}.
  - Upon receiving the summaries from the research assistants, review and analyze the content to identify key insights, trends, and connections.
  - Organize the information in a logical and coherent manner, ensuring that the report flows smoothly and covers all relevant aspects of the topic.
  - Include references to the original resources in the report, following a consistent citation style.
  - Proofread and edit the report for clarity, grammar, and formatting before considering it complete."""),
  expected_output="A comprehensive research report.",
  output_file='research_report.md', 
  agent=researcher
)

summarize_task = Task(
  description=dedent(f"""In-depth Reading and summarization. Topic: {topic}
  - Carefully review the instructions and clarify any doubts or questions with the Senior Researcher before proceeding.
  - Extract the content with the scrape tool
  - Thoroughly read and analyze the assigned resources, paying close attention to the key points, arguments, and evidence presented.
  - Identify the main themes, concepts, and ideas that are relevant to the research topic.
  - Craft concise and informative summaries of the assigned resources, capturing the essential information and insights.
  - Provide context and background information where appropriate to enhance the understanding of the summarized content.
  """),
  expected_output="A comprehensive summary of the given resources",
  output_file='resource_summaries.md',
  agent=resource_summarizer
)


# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, resource_summarizer],
  tasks=[websearch_task, delegation_task, summarize_task, report_task],
  verbose=1
)

# llm_func.with_structured_output()
# Get your crew to work!
result = crew.kickoff()
print(result)
