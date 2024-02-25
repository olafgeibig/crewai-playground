import os
from crewai import Crew
from tasks import GameTasks
from agents import GameAgents

os.environ["OPENAI_API_KEY"]="YOUR_API_KEY"

tasks = GameTasks()
agents = GameAgents()

print("## Welcome to the Game Crew")
print('-------------------------------')
game = "space shooter game, where the player can shoot at enemies and the enemies can shoot back, enemies come top to bottom at increasing speeds, if thei hit the player it's game over. There should be a score. Don't use any external assets, only code"

# Create Agents
senior_engineer_agent = agents.senior_engineer_agent()
qa_engineer_agent = agents.qa_engineer_agent()
chief_qa_engineer_agent = agents.chief_qa_engineer_agent()

# Create Tasks
code_game = tasks.code_task(senior_engineer_agent, game)
review_game = tasks.review_task(qa_engineer_agent, game)
approve_game = tasks.evaluate_task(chief_qa_engineer_agent, game)

# Create Crew responsible for Copy
crew = Crew(
	agents=[
		senior_engineer_agent,
		qa_engineer_agent,
		chief_qa_engineer_agent
	],
	tasks=[
		code_game,
		review_game,
		approve_game
	],
	verbose=True
)

game = crew.kickoff()