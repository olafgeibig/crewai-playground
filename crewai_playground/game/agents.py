from textwrap import dedent
from crewai import Agent
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI

class GameAgents():

	def senior_engineer_agent(self):
		llm = ChatOllama(
            model="olafgeibig/nous-hermes-2-mistral:7B-DPO-Q5_K_M",
            callbacks=CallbackManager([StreamingStdOutCallbackHandler()]),
        )
		# llm = ChatOpenAI(
        #     model="NousResearch/Nous-Hermes-2-Mistral-7B-DPO", 
        #     api_key="b4884adff34ff7665f0ca99cef4891db99ef6a33241f61f14bc6d4ff7609c542",
        #     base_url="https://api.together.xyz",
        #     callbacks=CallbackManager([StreamingStdOutCallbackHandler()]),
        #     streaming=True
        # )
		return Agent(
			role='Senior Software Engineer',
			goal='Create software as needed',
			backstory=dedent("""\
				You are a Senior Software Engineer at a leading tech think tank.
				Your expertise in programming in python. and do your best to
				produce perfect code"""),
			allow_delegation=False,
			llm=llm,
			verbose=True
		)

	def qa_engineer_agent(self):
		llm = ChatOllama(
            model="olafgeibig/nous-hermes-2-mistral:7B-DPO-Q5_K_M",
            callbacks=CallbackManager([StreamingStdOutCallbackHandler()]),
        )
		# llm = ChatOpenAI(
        #     model="NousResearch/Nous-Hermes-2-Mistral-7B-DPO", 
        #     api_key="b4884adff34ff7665f0ca99cef4891db99ef6a33241f61f14bc6d4ff7609c542",
        #     base_url="https://api.together.xyz",
        #     callbacks=CallbackManager([StreamingStdOutCallbackHandler()]),
        #     streaming=True
        # )
		return Agent(
			role='Software Quality Control Engineer',
  		goal='create prefect code, by analizing the code that is given for errors',
  		backstory=dedent("""\
				You are a software engineer that specializes in checking code
  			for errors. You have an eye for detail and a knack for finding
				hidden bugs.
  			You check for missing imports, variable declarations, mismatched
				brackets and syntax errors.
  			You also check for security vulnerabilities, and logic errors"""),
			allow_delegation=False,
			llm=llm,
			verbose=True
		)

	def chief_qa_engineer_agent(self):
		llm = ChatOllama(
            model="olafgeibig/nous-hermes-2-mistral:7B-DPO-Q5_K_M",
            callbacks=CallbackManager([StreamingStdOutCallbackHandler()]),
        )
		# llm = ChatOpenAI(
        #     model="NousResearch/Nous-Hermes-2-Mistral-7B-DPO", 
        #     api_key="b4884adff34ff7665f0ca99cef4891db99ef6a33241f61f14bc6d4ff7609c542",
        #     base_url="https://api.together.xyz",
        #     callbacks=CallbackManager([StreamingStdOutCallbackHandler()]),
        #     streaming=True
        # )
		return Agent(
			role='Chief Software Quality Control Engineer',
  		goal='Ensure that the code does the job that it is supposed to do',
  		backstory=dedent("""\
				You feel that programmers always do only half the job, so you are
				super dedicate to make high quality code."""),
			allow_delegation=True,
			llm=llm,
			verbose=True
		)