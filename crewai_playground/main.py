from crewai import Crew
from textwrap import dedent

from travel_agents import TravelAgents
from travel_tasks import TravelTasks

from dotenv import load_dotenv
load_dotenv()

class TripCrew:
  def __init__(self, origin, cities, date_range, interests):
    self.cities = cities
    self.origin = origin
    self.interests = interests
    self.date_range = date_range

  def run(self):
    agents = TravelAgents()
    tasks = TravelTasks()

    travel_reviewer = agents.travel_reviewer()
    local_expert_agent = agents.local_expert()
    travel_concierge_agent = agents.travel_concierge()

    # identify_task = tasks.identify_task(
    #   city_selector_agent,
    #   self.origin,
    #   self.cities,
    #   self.interests,
    #   self.date_range
    # )
    gather_task = tasks.gather_task(
      local_expert_agent,
      self.origin,
      self.cities,
      self.interests,
      self.date_range
    )
    plan_task = tasks.plan_task(
      travel_concierge_agent, 
      self.origin,
      self.interests,
      self.date_range
    )
    review_task = tasks.review_task(
      travel_reviewer, 
      self.origin,
      self.interests,
      self.date_range
    )

    crew = Crew(
      agents=[
        travel_reviewer, local_expert_agent, travel_concierge_agent
      ],
      tasks=[gather_task, plan_task, review_task],
      verbose=True
    )

    result = crew.kickoff()
    return result

if __name__ == "__main__":
  print("## Welcome to Trip Planner Crew")
  print('-------------------------------')
  location = input(
    dedent("""
      From where will you be travelling from?
    """))
  cities = input(
    dedent("""
      Where you want to travel?
    """))
  date_range = input(
    dedent("""
      What is the date range you are intereseted in traveling?
    """))
  interests = input(
    dedent("""
      What are some of your high level interests and hobbies?
    """))
  
  trip_crew = TripCrew(location, cities, date_range, interests)
  result = trip_crew.run()
  print("\n\n########################")
  print("## Here is you Trip Plan")
  print("########################\n")
  print(result)
