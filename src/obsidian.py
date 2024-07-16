from crewai import Crew, Agent, Task
from src.utils.llms import LLMFactory
from textwrap import dedent
from langchain_core.pydantic_v1 import BaseModel, Field
import os
from dotenv import load_dotenv
load_dotenv()

model = LLMFactory().get_anthropic_llm("claude-3-haiku-20240307")

agent = Agent(
  role='Data Analyst',
  goal='Extract actionable insights',
  backstory="""You're a data analyst at a large company.
  You're responsible for analyzing data and providing insights
  to the business.
  You're currently working on a project to analyze the
  performance of our marketing campaigns.""",
  tools=[],  # Optional, defaults to an empty list
  llm=model,)