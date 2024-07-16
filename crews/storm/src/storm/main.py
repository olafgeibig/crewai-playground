#!/usr/bin/env python
from storm.crew import StormCrew
from dotenv import load_dotenv

def run():
    load_dotenv()
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    inputs = {
        'topic': 'Autonomous agents driven by LLMs'
    }
    StormCrew().crew().kickoff(inputs=inputs)