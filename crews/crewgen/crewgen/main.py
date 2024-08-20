from crewgen.crew import CrewGenCrew
import sys

inputs = {
#     'crew_purpose': """
# Create a note about a given software project or product. research the internet about the project, identify 
# the links to the official resources, read the official resources and write a description of the project 
# and its concepts. Add a resources section with links to discussions, articles, blog posts, how-tos, videos, etc."""
    'crew_purpose': """
Take open github issues of a project, classify them as bugs, feature-requests, improvements, etc.
Understand the code base and judge the effort to work on each issue, classify them as easy, medium, hard, etc.
create a report with a visual representation of the issues and their properties, so that it's more easy to
decide which issue to work on next.""",
}

def run():
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    CrewGenCrew().crew().kickoff(inputs=inputs)

def train():
    """
    Train the crew for a given number of iterations.
    """
    try:
        CrewGenCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        CrewGenCrew().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    try:
        CrewGenCrew().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")
