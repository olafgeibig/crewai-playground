#!/usr/bin/env python
from article.crew import ArticleCrew


def run():
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    inputs = {
        'topic': 'Autonomous agents driven by LLMs'
    }
    ArticleCrew().crew().kickoff(inputs=inputs)