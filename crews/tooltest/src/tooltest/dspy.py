import os
from dotenv import load_dotenv
import dspy

# Load environment variables from .env file
load_dotenv()

# Define a prompt template for improving user prompts
class PromptImprover(dspy.Signature):
    """Improve a given user prompt."""

    input_prompt = dspy.InputField()
    improved_prompt = dspy.OutputField(desc="An improved version of the input prompt")

# Set up the language model
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

lm = dspy.OpenAI(api_key=api_key, model="gpt-3.5-turbo")
dspy.settings.configure(lm=lm)

# Create a teleprompter using the PromptImprover signature
improver = dspy.Teleprompter(PromptImprover)

def improve_prompt(user_prompt):
    """
    Improve a given user prompt using DSPy.
    
    Args:
    user_prompt (str): The original user prompt
    
    Returns:
    str: The improved prompt
    """
    result = improver(input_prompt=user_prompt)
    return result.improved_prompt

# Example usage
if __name__ == "__main__":
    original_prompt = "Write a story about a dog"
    improved_prompt = improve_prompt(original_prompt)
    
    print(f"Original prompt: {original_prompt}")
    print(f"Improved prompt: {improved_prompt}")